"""Same-origin authenticated speech UI and bounded WebSocket turn protocol."""
from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
import secrets
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from flask import jsonify, render_template, request, session

from p10_gateway import FlaskASGIAdapter
from web_service import create_authenticated_app
from dialogue.voices import voice_catalog


class DialogueState:
    def __init__(self, root, supervisor):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.supervisor = supervisor
        self.max_upload_mib = 1
        self.token = secrets.token_urlsafe(32)
        token_path = self.root / 'access-token'
        with token_path.open('x') as stream:
            token_path.chmod(0o600)
            stream.write(self.token+'\n')
        self.lock = threading.RLock()
        self.tickets = {}
        self.active = None

    def health(self):
        return self.supervisor.health()

    def ticket(self, origin):
        with self.lock:
            if self.active:
                raise ValueError('已有对话页面连接，请先断开那个页面')
            now = time.monotonic()
            self.tickets = {key:row for key,row in self.tickets.items() if row['expires'] > now}
            if len(self.tickets) >= 32:
                raise ValueError('连接请求过多，请一分钟后重试')
            key, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            self.tickets[key] = dict(token=token, origin=origin.rstrip('/'), expires=now+60)
            return dict(ws_path=f'/dialogue/ws/{key}', ws_token=token)

    def claim(self, key, token, origin):
        with self.lock:
            row = self.tickets.get(key)
            if (self.active or row is None or row['expires'] < time.monotonic()
                    or origin != row['origin'] or not secrets.compare_digest(token, row['token'])):
                return False
            del self.tickets[key]
            self.active = key
            return True


def decode_command(payload, turn_id):
    if not isinstance(payload, dict):
        raise ValueError('Expected an event object')
    kind = payload.get('type')
    if kind == 'begin':
        voice = payload.get('voice_id')
        if voice not in {row['id'] for row in voice_catalog()['voices']}:
            raise ValueError('请选择已校验的女声音色')
        request_id = payload.get('request_id')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
            raise ValueError('Invalid begin request ID')
        return dict(type=kind, voice_id=voice, request_id=request_id)
    if kind == 'attach':
        return dict(type=kind)
    if kind not in ('input', 'end', 'cancel', 'ack'):
        raise ValueError('Unknown event type')
    if not turn_id or payload.get('turn_id') != turn_id:
        raise ValueError('旧轮次已结束，请重新开始录音')
    command = dict(type=kind, turn_id=turn_id)
    if kind == 'input':
        sequence = payload.get('sequence')
        encoded = payload.get('pcm')
        if type(sequence) is not int or not 0 <= sequence <= 150:
            raise ValueError('Invalid PCM sequence')
        if not isinstance(encoded, str) or len(encoded) > 4268:
            raise ValueError('Each packet must contain at most 100 ms PCM')
        pcm = base64.b64decode(encoded, validate=True)
        if not 2 <= len(pcm) <= 3200 or len(pcm) % 2:
            raise ValueError('Expected mono 16 kHz PCM16')
        command.update(sequence=sequence, pcm=pcm)
    elif kind == 'ack':
        last = payload.get('last_audio_sequence')
        # Each TTS phrase can end in a short packet; the 15-second sample
        # budget does not imply at most 150 packet boundaries.
        if type(last) is not int or not 0 <= last <= 4095:
            raise ValueError('Invalid playback acknowledgement')
        command['last_audio_sequence'] = last
    return command


def create_dialogue_app(state, *, gateway_policy=None):
    http = create_authenticated_app(state)
    if gateway_policy is not None:
        http.config['SESSION_COOKIE_SECURE'] = True

    @http.get('/')
    @http.get('/dialogue')
    def dialogue_page():
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_urlsafe(24)
        catalog = voice_catalog()
        return render_template('dialogue.html', csrf_token=session['csrf_token'],
                               default_voice=catalog['default_voice'], voices=catalog['voices'])

    @http.get('/healthz')
    def health():
        data = state.health()
        return jsonify(data), 200 if data['status'] == 'ready' else 503

    @http.post('/dialogue/connect')
    def connect():
        if state.health()['status'] != 'ready':
            return jsonify(message='语音模型正在启动或恢复，请稍候'), 503
        try:
            return jsonify(state.ticket(request.host_url)), 201
        except ValueError as error:
            return jsonify(message=str(error)), 409

    @http.post('/dialogue/recover')
    def recover():
        state.supervisor.recover()
        return jsonify(status='recovering', history_cleared=True), 202

    @http.post('/dialogue/metrics')
    def metrics():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or data.get('kind') not in ('turn', 'interrupt'):
            return jsonify(message='测量格式无效'), 400
        clean = dict(kind=data['kind'], received_at=time.time())
        for name in ('stop_to_first_audio_s', 'media_seconds', 'play_wall_seconds',
                     'starvation_seconds', 'interrupt_stop_ms', 'sample_rate',
                     'background_count', 'packet_count'):
            value = data.get(name)
            if value is not None:
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 86400:
                    return jsonify(message='测量数值无效'), 400
                clean[name] = value
        for name in ('turn_id', 'voice_id', 'end_mode'):
            value = data.get(name)
            if value is not None:
                if not isinstance(value, str) or len(value) > 64:
                    return jsonify(message='测量标识无效'), 400
                clean[name] = value
        clean['complete'] = data.get('complete') is True
        with state.lock:
            with (state.root/'browser-metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(clean, ensure_ascii=False)+'\n')
        return jsonify(status='saved')

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    if gateway_policy is not None:
        from dialogue.public_gateway import PublicGatewayMiddleware
        app.add_middleware(PublicGatewayMiddleware, policy=gateway_policy)

    @app.websocket('/dialogue/ws/{key}')
    async def dialogue_socket(websocket: WebSocket, key: str):
        protocols = websocket.headers.get('sec-websocket-protocol', '').split(',')
        protocols = [row.strip() for row in protocols]
        if (len(protocols) != 2 or protocols[0] != 'liveact-dialogue-v1'
                or not state.claim(key, protocols[1], websocket.headers.get('origin', ''))):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol='liveact-dialogue-v1')
        current = {'turn_id':None, 'generation':state.supervisor.generation}

        async def receive_commands():
            while True:
                message = await websocket.receive_text()
                if len(message) > 8192:
                    raise ValueError('Event exceeds the packet budget')
                payload = json.loads(message)
                try:
                    command = decode_command(payload, current['turn_id'])
                    if command['type'] == 'begin':
                        current['turn_id'] = None
                    state.supervisor.send(command)
                except (ValueError, TypeError) as error:
                    await websocket.send_json(dict(type='command_error', message=str(error)))

        async def send_events():
            # Attaching returns bounded acknowledged history, without reloading
            # models. A disconnect cancels unfinished work and preserves only
            # previously acknowledged answers.
            state.supervisor.send(dict(type='attach'))
            while True:
                event = state.supervisor.next_event()
                if event is None:
                    await asyncio.sleep(.01)
                    continue
                if event['type'] == 'begun':
                    current['turn_id'] = event['turn_id']
                elif event['type'] in ('reset', 'service_error'):
                    current['turn_id'] = None
                elif event.get('turn_id') and event['turn_id'] != current['turn_id']:
                    continue
                # Lock generation check immediately before sending; stale
                # media from a replaced process cannot become a new answer.
                if event['backend_generation'] != state.supervisor.generation:
                    continue
                if 'pcm' in event:
                    event = dict(event, pcm=base64.b64encode(event['pcm']).decode('ascii'))
                await websocket.send_json(event)

        tasks = [asyncio.create_task(receive_commands()), asyncio.create_task(send_events())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if current['turn_id']:
                try:
                    state.supervisor.send(dict(type='cancel', turn_id=current['turn_id']))
                except ValueError:
                    pass
            with state.lock:
                if state.active == key:
                    state.active = None
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    app.mount('/', FlaskASGIAdapter(http))
    return app, http
