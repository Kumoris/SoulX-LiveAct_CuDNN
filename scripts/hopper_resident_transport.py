"""Private local transport for one resident renderer and bounded live turns."""
import base64
import copy
from functools import lru_cache
import hmac
import json
from pathlib import Path
import queue
import re
import secrets
import socket
import socketserver
import threading
import time
import wave

from src.audio_analysis.open_pcm import OpenPCMQueue, TurnCancelled


def write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


class Turn:
    def __init__(self, root, identifier, template, timeout):
        if not re.fullmatch(r'[a-z][a-z0-9-]{1,31}',identifier):
            raise ValueError('Safe unique turn ID required')
        self.id=identifier
        self.directory=Path(root)/'tasks'/identifier
        self.directory.mkdir(parents=True,exist_ok=False)
        self.data=copy.deepcopy(template)
        self.data['audio']=str(self.directory/'audio-3-mono16k.wav')
        self.queue=OpenPCMQueue(timeout=timeout)
        self.lock=threading.Lock()
        self.raw=(self.directory/'audio.pcm').open('wb',buffering=0)
        self.sequence=0
        self.status='queued'
        self.created_at=time.monotonic()
        self.eos_at=None
        self.cancel_at=None
        self.chunks=[]

    def push(self, sequence, packet):
        with self.lock:
            if sequence!=self.sequence:
                raise ValueError('PCM sequence must be contiguous')
            self.queue.push(packet)
            self.raw.write(packet)
            self.sequence+=1

    def finish(self):
        with self.lock:
            if self.queue.report()['received_samples']<8000:
                raise ValueError('At least 0.5 seconds of input required')
            if self.raw.closed:
                raise ValueError('Duplicate EOS or closed turn')
            self.raw.close()
            with wave.open(self.data['audio'],'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000)
                with (self.directory/'audio.pcm').open('rb') as source:
                    while packet:=source.read(65536):output.writeframesraw(packet)
            self.eos_at=time.monotonic()
            # The renderer learns the final sample count only at this EOS.
            self.queue.finish()

    def cancel(self, reason='Turn cancelled'):
        self.cancel_at=time.monotonic()
        self.queue.cancel(reason)
        with self.lock:
            if not self.raw.closed:self.raw.close()

    def snapshot(self):
        return dict(turn_id=self.id,status=self.status,created_at=self.created_at,
                    eos_at=self.eos_at,cancel_at=self.cancel_at,
                    chunks=[] if self.queue.cancelled else list(self.chunks),
                    audio=self.queue.report(),declared_total_samples=None)


class Coordinator:
    def __init__(self, root, template, *, input_timeout=15):
        self.root=Path(root).resolve();self.template=template
        self.pending=queue.Queue(maxsize=1)
        self.lock=threading.Lock()
        self.turns={}
        self.current=None
        self.input_timeout=input_timeout
        self.closed=threading.Event()
        self.model_ready=False
        self.model_loads=0
        self.token=secrets.token_urlsafe(32)
        token=self.root/'access-token';token.touch(mode=0o600,exist_ok=False)
        token.write_text(self.token+'\n')
        coordinator=self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(25)
                try:
                    raw=self.rfile.readline(20000)
                    if not raw.endswith(b'\n'):raise ValueError('Bounded JSON line required')
                    request=json.loads(raw)
                    if not hmac.compare_digest(str(request.pop('token','')),coordinator.token):
                        raise ValueError('Invalid private access token')
                    result=coordinator.dispatch(request)
                    response=dict(ok=True,result=result)
                except Exception as error:
                    response=dict(ok=False,error=f'{type(error).__name__}: {error}')
                self.wfile.write((json.dumps(response)+'\n').encode())

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address=False
            daemon_threads=True
            request_queue_size=8
        self.server=Server(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.save_service()

    def service_snapshot(self):
        return dict(status='ready' if self.model_ready else 'loading',
            updated_at=time.time(),host='127.0.0.1',port=self.server.server_address[1],
            model_loads=self.model_loads,world_size=2,protocol='private_json_pcm_prefix_eos',
            max_active_turns=1,max_pending_turns=1,max_audio_seconds=15,
            current_turn=self.current.id if self.current else None,
            browser_service_started=False,real_time_accepted=False)

    def save_service(self):
        write(self.root/'service.json',self.service_snapshot())

    def dispatch(self, request):
        action=request.get('action')
        if action=='status':
            identifier=request.get('turn_id')
            if identifier:
                if not re.fullmatch(r'[a-z][a-z0-9-]{1,31}',str(identifier)):
                    raise ValueError('Safe turn ID required')
                with self.lock:turn=self.turns.get(identifier)
                if turn is not None:return turn.snapshot()
                path=self.root/'tasks'/str(identifier)/'turn.json'
                if path.is_file():return json.loads(path.read_text())
                raise ValueError('Unknown turn')
            return self.service_snapshot()
        if action=='start':
            if self.closed.is_set() or not self.model_ready:
                raise ValueError('Renderer is not ready')
            if set(request)-{'action','turn_id'}:
                raise ValueError('No future sample count, waveform or duration may be declared')
            with self.lock:
                if self.pending.full():raise BufferError('One pending turn is already queued')
                turn=Turn(self.root,request['turn_id'],self.template,self.input_timeout)
                self.turns[turn.id]=turn
                self.pending.put_nowait(turn)
            return turn.snapshot()
        if action=='shutdown':
            self.closed.set()
            with self.lock:
                for turn in list(self.turns.values()):
                    if turn.status not in ('completed','cancelled'):turn.cancel('Renderer shutdown')
            return dict(status='shutdown_requested')
        with self.lock:turn=self.turns.get(request.get('turn_id'))
        if turn is None:raise ValueError('Unknown or retired turn')
        if action=='pcm':
            encoded=request.get('pcm','')
            if not isinstance(encoded,str) or len(encoded)>8600:raise ValueError('Bounded PCM packet required')
            turn.push(request['sequence'],base64.b64decode(encoded,validate=True))
        elif action=='eos':turn.finish()
        elif action=='cancel':turn.cancel()
        else:raise ValueError('Unknown resident action')
        return turn.snapshot()

    def next_turn(self):
        while not self.closed.is_set():
            try:
                turn=self.pending.get(timeout=.1)
                self.current=turn;turn.status='running';self.save_service()
                return turn
            except queue.Empty:pass
        return None

    def finish_turn(self, report):
        turn=self.current
        turn.status=report['status']
        value=dict(**turn.snapshot(),render=report)
        write(turn.directory/'turn.json',value)
        if not turn.raw.closed:turn.raw.close()
        with self.lock:self.turns.pop(turn.id,None)
        self.current=None
        self.save_service()

    def close(self):
        self.closed.set()
        for turn in list(self.turns.values()):
            if turn.status not in ('completed','cancelled'):turn.cancel('Renderer stopped')
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


@lru_cache(maxsize=8)
def endpoint(root):
    """Bootstrap once; live packets never depend on shared-FS status renames."""
    root=Path(root)
    deadline=time.monotonic()+2
    while True:
        try:
            service=json.loads((root/'service.json').read_text())
            token=(root/'access-token').read_text().strip()
            return service['host'],service['port'],token
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=deadline:raise
            time.sleep(.01)


def request(root, action, *, timeout=25, **payload):
    host,port,token=endpoint(str(Path(root).resolve()))
    value=dict(action=action,token=token,**payload)
    with socket.create_connection((host,port),timeout=timeout) as channel:
        channel.settimeout(timeout)
        channel.sendall((json.dumps(value)+'\n').encode())
        reader=channel.makefile('rb')
        data=reader.readline(100000)
    result=json.loads(data)
    if not result['ok']:raise RuntimeError(result['error'])
    return result['result']
