"""One resident CPU conversation, bounded streaming output and turn cancellation.

Native ASR/TTS calls may finish their current call before releasing the worker.
Epoch invalidation and playback cancellation do not wait for that call. A timed
out turn is failed and its buffers discarded; process supervision belongs to
the service if a native backend never returns.
"""
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field
import queue
import threading
import time
import uuid

from dialogue.local_backends import sentences


class TurnStopped(Exception):
    pass


@dataclass
class Turn:
    session_id: str
    turn_id: str
    epoch: int
    voice_id: str
    events: queue.Queue
    status: str = 'recording'
    input_pcm: bytearray = field(default_factory=bytearray)
    next_input_sequence: int = 0
    audio_packets: int = 0
    audio_samples: int = 0
    transcript: str = ''
    reply: str = ''
    context_messages: int = 0
    queue_high_water: int = 0
    error: str | None = None
    terminal_event: dict | None = None
    terminal_delivered: bool = False
    complete_delivered: bool = False
    stopped: threading.Event = field(default_factory=threading.Event)
    work_done: threading.Event = field(default_factory=threading.Event)
    timer: threading.Timer | None = None
    timer_generation: int = 0
    started_at: float = field(default_factory=time.monotonic)


class DialogueSession:
    sample_rate = 16000

    def __init__(self, asr, llm, tts, *, queue_events=16, max_input_seconds=15,
                 max_reply_seconds=15, recording_timeout=30, turn_timeout=45,
                 playback_timeout=30, max_history_pairs=4,
                 allowed_voice_ids=None):
        if queue_events < 2 or min(recording_timeout, turn_timeout, playback_timeout) <= 0:
            raise ValueError('Use positive deadlines and at least two output slots')
        if not 0.5 <= max_input_seconds <= 15 or not 0.5 <= max_reply_seconds <= 15:
            raise ValueError('Keep each input and spoken reply within 0.5–15 seconds')
        if not 1 <= max_history_pairs <= 8:
            raise ValueError('Conversation history must have a bounded pair count')
        self.session_id = uuid.uuid4().hex
        self.asr, self.llm, self.tts = asr, llm, tts
        self.queue_events = queue_events
        self.max_input_bytes = int(max_input_seconds*self.sample_rate)*2
        self.max_reply_samples = int(max_reply_seconds*self.sample_rate)
        self.recording_timeout, self.turn_timeout = recording_timeout, turn_timeout
        self.playback_timeout = playback_timeout
        if allowed_voice_ids is None:
            from dialogue.voices import voice_catalog
            allowed_voice_ids = [row['id'] for row in voice_catalog()['voices']]
        self.allowed_voice_ids = frozenset(allowed_voice_ids)
        self._history = deque(maxlen=max_history_pairs)
        self._condition = threading.Condition(threading.RLock())
        self._pending = queue.Queue(maxsize=1)
        self._current = None
        self._epoch = 0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name='local-dialogue-worker', daemon=True)
        self._worker.start()

    def _event(self, turn, kind, **payload):
        return dict(session_id=self.session_id, turn_id=turn.turn_id,
                    epoch=turn.epoch, type=kind, **payload)

    def _check(self, turn):
        if (self._closed or self._current is not turn or turn.epoch != self._epoch
                or turn.stopped.is_set()):
            raise TurnStopped('Turn was cancelled, replaced or closed')

    def _arm(self, turn, seconds, reason):
        if turn.timer:
            turn.timer.cancel()
        turn.timer_generation += 1
        turn.timer = threading.Timer(seconds, self._expire, (turn, reason, turn.timer_generation))
        turn.timer.daemon = True
        turn.timer.start()

    def _discard(self, turn, status, reason):
        turn.stopped.set()
        turn.status, turn.error = status, reason
        turn.input_pcm.clear()
        if turn.timer:
            turn.timer.cancel()
        while True:
            try:
                turn.events.get_nowait()
            except queue.Empty:
                break
        turn.terminal_event = self._event(turn, 'cancelled' if status == 'cancelled' else 'error', reason=reason)
        self._condition.notify_all()

    def _expire(self, turn, reason, generation):
        with self._condition:
            if (generation == turn.timer_generation and self._current is turn
                    and not turn.stopped.is_set() and turn.status != 'completed'):
                self._epoch += 1
                self._discard(turn, 'failed', reason)

    def begin_input(self, *, voice_id=None):
        voice_id = self.tts.voice_id if voice_id is None else voice_id
        if voice_id not in self.allowed_voice_ids:
            raise ValueError('Choose a prepared local voice ID')
        with self._condition:
            if self._closed:
                raise RuntimeError('Session is closed')
            if self._current and self._current.status != 'completed' and not self._current.stopped.is_set():
                self._discard(self._current, 'cancelled', 'replaced_by_new_turn')
            self._epoch += 1
            turn = Turn(self.session_id, uuid.uuid4().hex, self._epoch, voice_id,
                        queue.Queue(maxsize=self.queue_events))
            self._current = turn
            # Remove an obsolete queued request, leaving at most the current
            # native call plus one new request in memory.
            while True:
                try:
                    previous = self._pending.get_nowait()
                    if previous is not None:
                        previous[0].work_done.set()
                except queue.Empty:
                    break
            self._arm(turn, self.recording_timeout, 'recording_eos_timeout')
            return turn

    def append_pcm(self, turn, pcm, *, sequence):
        with self._condition:
            self._check(turn)
            if turn.status != 'recording':
                raise ValueError('Input is already sealed')
            reason = None
            if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
                reason = 'input_requires_mono_pcm16_16khz'
            elif type(sequence) is not int or sequence != turn.next_input_sequence:
                reason = 'input_sequence_mismatch'
            elif len(turn.input_pcm) + len(pcm) > self.max_input_bytes:
                reason = 'input_duration_exceeded'
            if reason:
                self._epoch += 1
                self._discard(turn, 'failed', reason)
                raise ValueError(reason)
            turn.input_pcm.extend(pcm)
            turn.next_input_sequence += 1

    def end_input(self, turn):
        with self._condition:
            self._check(turn)
            if turn.status != 'recording':
                raise ValueError('EOS is only accepted once')
            if len(turn.input_pcm) < self.sample_rate:
                self._epoch += 1
                self._discard(turn, 'failed', 'input_shorter_than_half_second')
                raise ValueError('Input must contain at least 0.5 seconds')
            pcm = bytes(turn.input_pcm)
            turn.input_pcm.clear()
            turn.status = 'queued'
            self._arm(turn, self.turn_timeout, 'generation_or_consumer_timeout')
            self._pending.put_nowait((turn, pcm))

    def submit_audio(self, pcm, *, voice_id=None):
        turn = self.begin_input(voice_id=voice_id)
        self.append_pcm(turn, pcm, sequence=0)
        self.end_input(turn)
        return turn

    def cancel(self, turn, *, reason='user_interrupt'):
        with self._condition:
            if turn.session_id != self.session_id:
                raise ValueError('Turn belongs to another session')
            if self._current is turn and not turn.stopped.is_set() and turn.status != 'completed':
                self._epoch += 1
                self._discard(turn, 'cancelled', reason)
            return self._event(turn, 'cancelled', reason=turn.error or reason)

    def _publish(self, turn, kind, **payload):
        with self._condition:
            event = self._event(turn, kind, **payload)
            while True:
                self._check(turn)
                try:
                    turn.events.put_nowait(event)
                    turn.queue_high_water = max(turn.queue_high_water, turn.events.qsize())
                    self._condition.notify_all()
                    return
                except queue.Full:
                    self._condition.wait(0.05)

    def read_event(self, turn, *, timeout=0.1):
        deadline = time.monotonic() + timeout
        with self._condition:
            if turn.session_id != self.session_id:
                raise ValueError('Turn belongs to another session')
            while True:
                if turn.terminal_event:
                    if not turn.terminal_delivered:
                        turn.terminal_delivered = True
                        return dict(turn.terminal_event)
                    return None
                if self._current is not turn or turn.epoch != self._epoch:
                    return None
                try:
                    event = turn.events.get_nowait()
                    if event['type'] == 'complete':
                        turn.complete_delivered = True
                    self._condition.notify_all()
                    return event
                except queue.Empty:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or turn.status == 'completed':
                        return None
                    self._condition.wait(remaining)

    def acknowledge_playback(self, turn, *, last_audio_sequence):
        with self._condition:
            self._check(turn)
            if (turn.status != 'awaiting_playback' or not turn.complete_delivered
                    or type(last_audio_sequence) is not int
                    or last_audio_sequence != turn.audio_packets-1):
                raise ValueError('Acknowledge only the fully played current answer')
            if not turn.events.empty():
                raise ValueError('Answer events have not all been delivered')
            self._history.append((turn.transcript, turn.reply))
            turn.status = 'completed'
            if turn.timer:
                turn.timer.cancel()

    def history(self):
        with self._condition:
            return list(self._history)

    def snapshot(self, turn):
        with self._condition:
            return dict(session_id=turn.session_id, turn_id=turn.turn_id, epoch=turn.epoch,
                        status=turn.status, voice_id=turn.voice_id, transcript=turn.transcript,
                        reply=turn.reply, audio_packets=turn.audio_packets,
                        audio_samples=turn.audio_samples, context_messages=turn.context_messages,
                        queue_high_water=turn.queue_high_water, queue_capacity=self.queue_events,
                        buffered_input_bytes=len(turn.input_pcm), error=turn.error,
                        worker_released=turn.work_done.is_set())

    def _process(self, turn, pcm):
        with self._condition:
            self._check(turn)
            turn.status = 'recognizing'
        self._publish(turn, 'stage', stage='recognizing')
        transcription = self.asr.transcribe(pcm)
        with self._condition:
            self._check(turn)
            turn.transcript = transcription['text'].strip()
        self._publish(turn, 'recognized', **transcription)
        if not turn.transcript:
            with self._condition:
                self._check(turn)
                turn.status = 'completed'
                turn.timer.cancel()
            self._publish(turn, 'no_speech')
            return
        if len(turn.transcript) > 512:
            raise ValueError('Recognized text exceeds the single-turn text budget')
        with self._condition:
            messages = [dict(role='system', content='你是友好的中文数字人助手。结合上下文回答，优先用不超过四十个汉字的一句话，不要使用Markdown。')]
            for question, answer in self._history:
                messages.extend([dict(role='user', content=question), dict(role='assistant', content=answer)])
            messages.append(dict(role='user', content=turn.transcript))
            while sum(len(row['content']) for row in messages) > 1536 and len(messages) > 2:
                del messages[1:3]
            if hasattr(self.llm, 'prepare_context'):
                messages = self.llm.prepare_context(messages, max_tokens=128)
            turn.context_messages = len(messages)
            turn.status = 'answering'
        self._publish(turn, 'stage', stage='answering')
        reply = []
        with closing(self.llm.stream(messages, max_tokens=128)) as source:
            def tokens():
                for text in source:
                    with self._condition:
                        self._check(turn)
                    reply.append(text)
                    self._publish(turn, 'text', text=text)
                    yield text
            with closing(tokens()) as deltas:
                for phrase in sentences(deltas):
                    with self._condition:
                        self._check(turn)
                        turn.status = 'synthesizing'
                    self._publish(turn, 'stage', stage='synthesizing')
                    with closing(self.tts.stream(phrase, voice_id=turn.voice_id)) as audio:
                        for part in audio:
                            with self._condition:
                                self._check(turn)
                            if not isinstance(part, bytes) or not part or len(part) % 2:
                                raise ValueError('TTS requires finite non-empty PCM16 output')
                            if turn.audio_samples+len(part)//2 > self.max_reply_samples:
                                raise ValueError('Spoken reply exceeds the 15-second budget')
                            # Emit 100 ms packets: a slow consumer cannot retain
                            # whole unbounded sentences in its event queue.
                            for offset in range(0, len(part), 3200):
                                chunk = part[offset:offset+3200]
                                self._publish(turn, 'audio', sequence=turn.audio_packets,
                                              sample_start=turn.audio_samples, sample_rate=self.sample_rate,
                                              pcm=chunk)
                                turn.audio_packets += 1
                                turn.audio_samples += len(chunk)//2
        if not reply or not turn.audio_samples:
            raise ValueError('No spoken answer was generated')
        with self._condition:
            self._check(turn)
            turn.reply = ''.join(reply)
            turn.status = 'awaiting_playback'
            self._arm(turn, self.playback_timeout, 'playback_ack_timeout')
        self._publish(turn, 'complete', reply=turn.reply, audio_packets=turn.audio_packets,
                      audio_samples=turn.audio_samples)

    def _run(self):
        while True:
            work = self._pending.get()
            if work is None:
                return
            turn, pcm = work
            try:
                self._process(turn, pcm)
            except TurnStopped:
                pass
            except Exception as error:
                with self._condition:
                    if self._current is turn and not turn.stopped.is_set():
                        self._epoch += 1
                        self._discard(turn, 'failed', f'{type(error).__name__}: {error}')
            finally:
                turn.work_done.set()

    def close(self, *, timeout=3):
        with self._condition:
            self._closed = True
            self._epoch += 1
            if self._current and self._current.status != 'completed' and not self._current.stopped.is_set():
                self._discard(self._current, 'cancelled', 'session_closed')
            while True:
                try:
                    previous = self._pending.get_nowait()
                    if previous is not None:
                        previous[0].work_done.set()
                except queue.Empty:
                    break
            self._pending.put_nowait(None)
            self._condition.notify_all()
        self._worker.join(timeout)
        return not self._worker.is_alive()
