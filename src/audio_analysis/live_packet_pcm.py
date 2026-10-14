"""Bounded packet source for isolated, wall-clock paced resident-worker tests.

Only the producer reads the WAV fixture. The conditioner can access packets
already delivered to this source; it never receives a full future waveform.
Declared duration is a test protocol constraint, not microphone EOS detection.
"""
import queue
import threading
import time
import wave

from src.audio_analysis.streaming_audio import ReplayPreparedPCM


class PacketBytes:
    def __init__(self, samples, *, capacity=64, timeout=15):
        if type(samples) is not int or samples < 8000:
            raise ValueError('At least 0.5 seconds must be declared')
        if type(capacity) is not int or capacity < 1 or timeout <= 0:
            raise ValueError('Invalid packet capacity or timeout')
        self.samples, self.timeout = samples, timeout
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()
        self.eos = threading.Event()
        self.lock = threading.Lock()
        self.received = self.consumed = self.peak_pending_samples = 0
        self.error = None
        self.started = self.eos_at = None

    def __len__(self):
        return self.samples * 2

    def push(self, packet, *, backpressure=False):
        if not isinstance(packet, bytes) or not packet or len(packet) % 2 or len(packet) > 6400:
            raise ValueError('Expected at most 200 ms of PCM16')
        if self.closed.is_set() or self.eos.is_set():
            raise RuntimeError('Packet stream is closed')
        deadline = time.monotonic() + self.timeout
        while True:
            if self.closed.is_set():
                raise RuntimeError('Packet stream was cancelled')
            with self.lock:
                if self.received + len(packet)//2 > self.samples:
                    raise ValueError('Input exceeds declared duration')
                try:
                    self.queue.put_nowait(packet)
                except queue.Full:
                    if not backpressure:
                        raise BufferError('Live input queue exceeded capacity; no audio was dropped')
                else:
                    self.received += len(packet)//2
                    self.peak_pending_samples = max(self.peak_pending_samples, self.received-self.consumed)
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError('Packet producer backpressure timeout')
            self.closed.wait(.01)

    def finish(self):
        if self.eos.is_set() or self.closed.is_set() or self.received != self.samples:
            raise ValueError('EOS must match the complete declared input exactly once')
        self.eos_at = time.monotonic()
        self.eos.set()

    def fail(self, error):
        self.error = error
        self.closed.set()

    def __getitem__(self, span):
        if (not isinstance(span, slice) or span.step is not None
                or span.start != self.consumed * 2
                or span.stop != min(len(self), span.start + 6400)):
            raise ValueError('Only sequential, complete 200 ms packets may be consumed')
        deadline = time.monotonic() + self.timeout
        while True:
            if self.error is not None:
                raise RuntimeError('Packet producer failed') from self.error
            if self.closed.is_set():
                raise RuntimeError('Packet stream was cancelled')
            try:
                packet = self.queue.get(timeout=.05)
                break
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Waiting for an audio packet timed out')
        if len(packet) != span.stop - span.start:
            raise ValueError('Packet size does not match its declared sequence')
        with self.lock:
            self.consumed += len(packet) // 2
        if self.consumed == self.samples:
            if not self.eos.wait(self.timeout):
                raise TimeoutError('Last packet arrived without explicit EOS')
        return packet

    def report(self):
        with self.lock:
            return dict(received_samples=self.received, consumed_samples=self.consumed,
                        pending_samples=self.received-self.consumed,
                        peak_pending_samples=self.peak_pending_samples,
                        eos_received=self.eos.is_set(), producer_started=self.started,
                        producer_eos_at=self.eos_at)

    def close(self):
        self.closed.set()


class LiveFixturePCM(ReplayPreparedPCM):
    """Use the accepted DSP/window code with a bounded producer packet queue."""
    def __init__(self, path, *, fps, sox, paced=False, capacity=64, start_on_window=False):
        with wave.open(str(path)) as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('The live wire format is mono PCM16 / 16 kHz')
            samples = wav.getnframes()
        # The base validates a minimum placeholder, then all reads use the
        # PacketBytes source below. No future waveform is passed to it.
        super().__init__(bytes(16000), fps=fps, sox=sox)
        self.source = PacketBytes(samples, capacity=capacity)
        self.paced = paced
        self.producer = threading.Thread(target=self._produce, args=(path,), daemon=True)
        self.started = False
        if not start_on_window:
            self.start()

    def start(self):
        if not self.started:
            self.started = True
            self.producer.start()

    def _produce(self, path):
        self.source.started = time.monotonic()
        try:
            with wave.open(str(path)) as wav:
                sent = 0
                while not self.source.closed.is_set():
                    packet = wav.readframes(3200)
                    if not packet:
                        self.source.finish()
                        return
                    sent += len(packet)//2
                    if self.paced:
                        delay = self.source.started + sent/16000 - time.monotonic()
                        if delay > 0 and self.source.closed.wait(delay):
                            return
                    self.source.push(packet, backpressure=not self.paced)
        except Exception as error:
            self.source.fail(error)

    def window(self, *args, **kwargs):
        self.start()
        window, pcm, metadata = super().window(*args, **kwargs)
        metadata.update(self.source.report(), input_wall_clock_paced=self.paced,
                        input_source='bounded_packet_queue',
                        mode_scope='Isolated prerecorded producer; not the browser microphone gateway')
        return window, pcm, metadata

    def close(self):
        self.source.close()
        if self.started:
            self.producer.join(timeout=2)
        super().close()
        if self.producer.is_alive():
            raise RuntimeError('PCM producer did not stop')
