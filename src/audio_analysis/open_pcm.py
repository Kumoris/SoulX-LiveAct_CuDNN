"""Bounded PCM without a declared future duration; explicit EOS seals it."""
from collections import deque
import math
import threading
import time

from src.audio_analysis.audio_window import audio_window
from src.audio_analysis.streaming_audio import ReplayPreparedPCM


class TurnCancelled(RuntimeError):
    pass


class OpenPCMQueue:
    def __init__(self, *, max_samples=240000, capacity_samples=204800, timeout=15):
        if max_samples < 8000 or capacity_samples < 3200 or timeout <= 0:
            raise ValueError('Invalid open PCM bounds')
        self.max_samples, self.capacity_samples, self.timeout = max_samples, capacity_samples, timeout
        self.condition = threading.Condition()
        self.packets = deque()
        self.pending = bytearray()
        self.received = self.consumed = self.peak_pending_samples = 0
        self.eos = False
        self.cancelled = False
        self.reason = None

    def push(self, packet):
        if not isinstance(packet, bytes) or not packet or len(packet) % 2 or len(packet) > 6400:
            raise ValueError('Mono PCM16 packet must contain 1--3200 samples')
        deadline = time.monotonic()+self.timeout
        with self.condition:
            if self.received+len(packet)//2 > self.max_samples:
                raise ValueError('Per-turn audio exceeds 15-second budget')
            while self.received-self.consumed+len(packet)//2 > self.capacity_samples:
                self._open()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('PCM producer backpressure timed out')
                self.condition.wait(min(.1, remaining))
            self._open()
            if self.received+len(packet)//2 > self.max_samples:
                raise ValueError('Per-turn audio exceeds 15-second budget')
            self.packets.append(packet)
            self.received += len(packet)//2
            self.peak_pending_samples = max(self.peak_pending_samples, self.received-self.consumed)
            self.condition.notify_all()

    def _open(self):
        if self.cancelled:
            raise TurnCancelled(self.reason or 'Turn cancelled')
        if self.eos:
            raise ValueError('PCM already reached EOS')

    def finish(self):
        with self.condition:
            self._open()
            if self.received < 8000:
                raise ValueError('At least 0.5 seconds required at EOS')
            self.eos = True
            self.condition.notify_all()

    def cancel(self, reason='Turn cancelled'):
        with self.condition:
            self.cancelled = True
            self.reason = reason
            self.packets.clear()
            self.pending.clear()
            self.condition.notify_all()

    def read_packet(self):
        """Preserve 200-ms DSP boundaries; a partial tail requires real EOS."""
        deadline = time.monotonic()+self.timeout
        with self.condition:
            while True:
                if self.cancelled:
                    raise TurnCancelled(self.reason or 'Turn cancelled')
                while self.packets and len(self.pending) < 6400:
                    self.pending.extend(self.packets.popleft())
                if len(self.pending) >= 6400 or self.eos:
                    size = min(6400, len(self.pending))
                    packet = bytes(self.pending[:size])
                    del self.pending[:size]
                    self.consumed += size//2
                    eof = self.eos and not self.pending and not self.packets
                    self.condition.notify_all()
                    return packet, eof
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self.cancelled = True
                    self.reason = 'Audio packet/EOS deadline exceeded'
                    self.condition.notify_all()
                    raise TurnCancelled(self.reason)
                self.condition.wait(min(.1, remaining))

    def report(self):
        with self.condition:
            return dict(received_samples=self.received, consumed_samples=self.consumed,
                pending_samples=0 if self.cancelled else self.received-self.consumed,
                peak_pending_samples=self.peak_pending_samples,
                capacity_samples=self.capacity_samples, eos_received=self.eos,
                cancelled=self.cancelled, declared_total_samples=None)


class OpenPreparedPCM(ReplayPreparedPCM):
    """Same tempo and audio-window math; only delivered prefix enters DSP."""
    def __init__(self, source, *, fps, sox):
        super().__init__(bytes(16000), fps=fps, sox=sox)
        self.source = source

    def window(self, condition_start, condition_end, *, history_seconds,
               extra_lookahead_seconds):
        tick = time.perf_counter()
        horizon = (condition_end+2+math.ceil(extra_lookahead_seconds*self.fps-1e-7))*640
        planned = audio_window(condition_start, condition_end,
            total_samples=max(horizon+640, self.prepared_samples), fps=self.fps,
            history_seconds=history_seconds, extra_lookahead_seconds=extra_lookahead_seconds)
        if planned.prepared_start_sample < self.buffer_start:
            raise ValueError('Requested PCM history was evicted')
        discard = min(self.buffer.size, planned.prepared_start_sample-self.buffer_start)
        self.buffer = self.buffer[discard:].copy()
        self.buffer_start += discard
        while self.prepared_samples < horizon and not self.source_eof:
            packet, eof = self.source.read_packet()
            if packet:
                self.preparer.push(packet)
                self.source_samples += len(packet)//2
                self._append(self.preparer.read_available(wait_seconds=.01))
            if eof:
                self.source_eof = True
                self._append(self.preparer.finish())
        total = self.prepared_samples if self.source_eof else max(horizon+640, self.prepared_samples)
        window = audio_window(condition_start, condition_end, total_samples=total,
            fps=self.fps, history_seconds=history_seconds, extra_lookahead_seconds=extra_lookahead_seconds)
        start, end = window.prepared_start_sample-self.buffer_start, window.prepared_end_sample-self.buffer_start
        if start < 0 or end > self.buffer.size:
            raise ValueError('Prepared live PCM window unavailable')
        metadata = dict(input_samples_received=self.source_samples,
            input_seconds_received=self.source_samples/16000,
            prepared_samples_emitted=self.prepared_samples,
            source_eof_received=self.source_eof,
            buffer_start_sample=self.buffer_start, retained_prepared_samples=int(self.buffer.size),
            peak_retained_prepared_samples=int(self.peak_buffer_samples),
            first_output_input_sample=self.first_output_input_sample,
            preparation_excess_input_seconds=max(0,self.source_samples/16000-window.prepared_end_sample*(25/self.fps)/16000),
            preparation_and_wait_wall_seconds=time.perf_counter()-tick,
            packet_samples=3200,sox_command=self.preparer.command,
            input_source='open_bounded_pcm_with_explicit_eos',declared_total_samples=None,
            queue=self.source.report())
        return window, self.buffer[start:end].copy(), metadata
