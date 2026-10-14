"""Bounded CPU worker process; a stuck native call cannot kill the web server."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

from dialogue.session import DialogueSession
from dialogue.voices import voice_catalog


def _models(root):
    from dialogue.local_backends import LocalASR, LocalLLM, LocalTTS
    root = Path(root) / 'models/dialogue'
    return (LocalASR(root / 'faster-whisper-small'),
            LocalLLM(root / 'Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf'),
            LocalTTS(root / 'Kokoro-82M-v1.1-zh'))


def _worker(root, generation, commands, media, control, heartbeat, recycle_at,
            backend_factory, warm, cancel_grace, session_options):
    # These queues have independent bounded capacities. A full media queue
    # must never block cancellation, heartbeat or the hard-recovery signal.
    media.cancel_join_thread()
    control.cancel_join_thread()
    session = None
    current, pending, lingering = None, None, []
    dispatched = set()
    def emit(kind, **payload):
        control.put(dict(type=kind, backend_generation=generation, **payload), timeout=1)
    try:
        start = time.monotonic()
        asr, llm, tts = (backend_factory(root) if backend_factory else _models(root))
        if warm:
            asr.transcribe(bytes(32000))
            source = llm.stream([dict(role='user', content='请说你好。')], max_tokens=8)
            try:
                list(source)
            finally:
                source.close()
            list(tts.stream('你好。', voice_id=tts.voice_id))
        session = DialogueSession(asr, llm, tts, **session_options)
        catalog = voice_catalog()
        def attached(kind='attached', **extra):
            emit(kind, session_id=session.session_id, default_voice=tts.voice_id,
                 voices=[row['id'] for row in catalog['voices']],
                 history=session.history(), **extra)
        attached('ready', model_load_and_warm_seconds=time.monotonic()-start)
        while True:
            now = time.monotonic()
            heartbeat.value = now
            lingering = [row for row in lingering if not row[0].work_done.is_set()]
            if current and current.stopped.is_set() and not current.work_done.is_set():
                # Recording turns never entered the native worker.
                if current.turn_id in dispatched:
                    if not any(row[0] is current for row in lingering):
                        lingering.append((current, now+cancel_grace))
            recycle_at.value = min((row[1] for row in lingering), default=0)
            try:
                command = commands.get(timeout=.01)
            except queue.Empty:
                command = None
            if command:
                kind = command['type']
                try:
                    if kind == 'shutdown':
                        break
                    if kind == 'attach':
                        attached()
                    elif kind == 'begin':
                        if current and current.status not in ('completed', 'recording') and not current.work_done.is_set():
                            lingering.append((current, now+cancel_grace))
                        current = session.begin_input(voice_id=command.get('voice_id'))
                        pending = None
                        emit('begun', request_id=command['request_id'], **session.snapshot(current))
                    else:
                        if current is None or command.get('turn_id') != current.turn_id:
                            raise ValueError('This turn is no longer current')
                        if kind == 'input':
                            session.append_pcm(current, command['pcm'], sequence=command['sequence'])
                        elif kind == 'end':
                            session.end_input(current)
                            dispatched = {current.turn_id}
                            emit('accepted', **session.snapshot(current))
                        elif kind == 'cancel':
                            was_native = current.status not in ('recording', 'completed')
                            event = session.cancel(current)
                            pending = None
                            emit('cancelled', **{k:v for k,v in event.items() if k != 'type'})
                            if was_native and not current.work_done.is_set():
                                lingering.append((current, now+cancel_grace))
                        elif kind == 'ack':
                            session.acknowledge_playback(current, last_audio_sequence=command['last_audio_sequence'])
                            emit('committed', **session.snapshot(current))
                        else:
                            raise ValueError('Unknown worker command')
                except Exception as error:
                    emit('command_error', message=str(error), turn_id=command.get('turn_id'),
                         request_id=command.get('request_id'))
            if current:
                if pending and (pending.get('turn_id') != current.turn_id or current.stopped.is_set()):
                    pending = None
                if pending is None:
                    pending = session.read_event(current, timeout=0)
                    if pending:
                        pending['backend_generation'] = generation
                if pending:
                    # Terminal controls bypass media backpressure.
                    if pending['type'] in ('error', 'cancelled', 'no_speech'):
                        control.put(pending, timeout=1)
                        pending = None
                    else:
                        try:
                            media.put_nowait(pending)
                            pending = None
                        except queue.Full:
                            pass
    except BaseException as error:
        try:
            emit('worker_error', message=f'{type(error).__name__}: {error}')
        except (queue.Full, OSError):
            pass
    finally:
        if session:
            session.close(timeout=.2)


class DialogueSupervisor:
    def __init__(self, root, *, backend_factory=None, warm=True, startup_timeout=420,
                 cancel_grace=8, session_options=None, max_restarts=3):
        self.root = str(root)
        self.backend_factory, self.warm = backend_factory, warm
        self.startup_timeout, self.cancel_grace = startup_timeout, cancel_grace
        self.session_options, self.max_restarts = session_options or {}, max_restarts
        self.ctx = mp.get_context('spawn')
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.notices = queue.Queue(maxsize=32)
        self.generation = 0
        self.restarts = 0
        self.phase, self.error = 'initializing', None
        self.ready_info = {}
        self.process = None
        self._restart_request = None
        self._spawn()
        self.monitor = threading.Thread(target=self._watch, name='dialogue-supervisor', daemon=True)
        self.monitor.start()

    def _spawn(self):
        self.generation += 1
        self.commands = self.ctx.Queue(maxsize=32)
        self.media = self.ctx.Queue(maxsize=16)
        self.control = self.ctx.Queue(maxsize=16)
        self.heartbeat = self.ctx.Value('d', time.monotonic())
        self.recycle_at = self.ctx.Value('d', 0)
        self.started_at = time.monotonic()
        self.phase, self.error, self.ready_info = 'initializing', None, {}
        self.process = self.ctx.Process(target=_worker, args=(
            self.root, self.generation, self.commands, self.media, self.control,
            self.heartbeat, self.recycle_at, self.backend_factory, self.warm,
            self.cancel_grace, self.session_options), daemon=True)
        self.process.start()

    def _notice(self, event):
        try:
            self.notices.put_nowait(event)
        except queue.Full:
            # A disappeared client cannot retain an unbounded control history.
            self.notices.get_nowait()
            self.notices.put_nowait(event)

    def _dispose(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1)
        for channel in (self.commands, self.media, self.control):
            channel.cancel_join_thread()
            channel.close()

    def _watch(self):
        while not self.stop.wait(.05):
            with self.lock:
                if self.phase == 'error':
                    if self._restart_request is None:
                        continue
                    self.restarts = 0
                reason = self._restart_request
                self._restart_request = None
                while True:
                    try:
                        event = self.control.get_nowait()
                    except (queue.Empty, ValueError, OSError):
                        break
                    if event['type'] == 'ready':
                        self.phase = 'ready'
                        self.ready_info = event
                    elif event['type'] == 'worker_error':
                        reason = event['message']
                    self._notice(event)
                now = time.monotonic()
                if reason is None and not self.process.is_alive():
                    reason = f'worker_exited_{self.process.exitcode}'
                if reason is None and self.phase == 'initializing' and now-self.started_at > self.startup_timeout:
                    reason = 'model_startup_timeout'
                if reason is None and self.phase == 'ready':
                    if self.recycle_at.value and now >= self.recycle_at.value:
                        reason = 'cancelled_native_call_did_not_release'
                    elif now-self.heartbeat.value > 15:
                        reason = 'worker_control_heartbeat_timeout'
                if reason:
                    self._dispose()
                    self.restarts += 1
                    if self.restarts > self.max_restarts:
                        self.phase, self.error = 'error', reason
                        self._notice(dict(type='service_error', message=reason, backend_generation=self.generation))
                    else:
                        self._spawn()
                        self._notice(dict(type='reset', reason=reason, history_cleared=True,
                                          backend_generation=self.generation))

    def health(self):
        with self.lock:
            return dict(status=self.phase, worker_alive=self.process.is_alive(),
                        backend_generation=self.generation, restarts=self.restarts,
                        default_voice=voice_catalog()['default_voice'], error=self.error,
                        avatar_connected=False, real_time_accepted=False)

    def send(self, command):
        with self.lock:
            if self.phase != 'ready':
                raise ValueError('语音模型正在启动或恢复，请稍候')
            try:
                self.commands.put_nowait(command)
            except queue.Full:
                raise ValueError('录音发送过快，请打断后重试')

    def next_event(self):
        with self.lock:
            for source in (self.notices, self.media):
                while True:
                    try:
                        event = source.get_nowait()
                    except (queue.Empty, OSError, ValueError):
                        break
                    if event.get('backend_generation') == self.generation:
                        return event
        return None

    def recover(self):
        with self.lock:
            self._restart_request = 'user_requested_recovery'

    def close(self):
        self.stop.set()
        self.monitor.join(2)
        with self.lock:
            self._dispose()
            self.phase = 'stopped'
