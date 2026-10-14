"""Opt-in, authenticated canonical HTTPS origin for the public application gateway.

The gateway terminates TLS, then uses private SSH forwarding. Forwarded headers
alone are never sufficient to change the browser origin or bypass application
login/CSRF checks. Existing localhost access has no gateway policy by default.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GatewayPolicy:
    origin: str
    host: str
    port: int
    key: bytes
    key_file: str

    @classmethod
    def load(cls, origin, key_file):
        parsed = urlsplit(origin)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
                or parsed.password or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment or parsed.port not in (None, 443)
                or not re.fullmatch(r'[a-zA-Z0-9.-]+', parsed.hostname)):
            raise ValueError('LIVEACT_PUBLIC_ORIGIN must be a root HTTPS domain on port 443')
        path = Path(key_file).resolve()
        if path.stat().st_mode & 0o077:
            raise ValueError('Gateway key file must be private (chmod 600)')
        key = path.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', key):
            raise ValueError('Gateway key must contain 64 lowercase hexadecimal characters')
        host = parsed.hostname.lower()
        return cls('https://' + host, host, 443, key.encode('ascii'), str(path))

    @classmethod
    def from_environment(cls):
        origin = os.environ.get('LIVEACT_PUBLIC_ORIGIN', '')
        key_file = os.environ.get('LIVEACT_GATEWAY_KEY_FILE', '')
        if not origin and not key_file:
            return None
        if not origin or not key_file:
            raise ValueError('Set both LIVEACT_PUBLIC_ORIGIN and LIVEACT_GATEWAY_KEY_FILE')
        return cls.load(origin, key_file)


class PublicGatewayMiddleware:
    def __init__(self, app, policy):
        self.app = app
        self.policy = policy

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        headers = scope.get('headers', [])
        keys = [value for name, value in headers if name.lower() == b'x-liveact-gateway-key']
        hosts = [value.decode('latin-1').lower() for name, value in headers if name.lower() == b'host']
        allowed_host = (self.policy.host, self.policy.host + ':443')
        valid = (len(keys) == 1 and secrets.compare_digest(keys[0], self.policy.key)
                 and len(hosts) == 1 and hosts[0] in allowed_host)
        if not valid:
            if scope['type'] == 'websocket':
                await send(dict(type='websocket.close', code=1008))
            else:
                await send(dict(type='http.response.start', status=403,
                                headers=[(b'content-type', b'application/json'),
                                         (b'cache-control', b'no-store')]))
                await send(dict(type='http.response.body',
                                body=b'{"message":"Gateway verification failed"}'))
            return
        trusted = dict(scope)
        trusted['scheme'] = 'wss' if scope['type'] == 'websocket' else 'https'
        trusted['server'] = (self.policy.host, self.policy.port)
        # The canonical host is configured, not copied from a client-controlled
        # X-Forwarded-* value. Remove internal credentials before dispatch.
        trusted['headers'] = [(name, value) for name, value in headers
                              if name.lower() not in (b'host', b'x-liveact-gateway-key')
                              and not name.lower().startswith(b'x-forwarded-')]
        trusted['headers'].append((b'host', self.policy.host.encode('ascii')))
        return await self.app(trusted, receive, send)
