"""Stateful SoX tempo for the existing mono PCM16 / 16 kHz wire format.

One subprocess retains overlap/search state across packets. Only packets
already pushed are visible to it; EOF is sent explicitly. Other sample rates
are rejected, because substituting SoX rate for TorchAudio resampling would
need a separate numerical and quality comparison.
"""

import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time


class StreamingTempoPCM:
    sample_rate = 16000
    max_packet_samples = 3200

    def __init__(self, *, fps=20, sox=None, max_pending_samples=128000,
                 io_timeout=10):
        if type(fps) is not int or not 1 <= fps <= 100:
            raise ValueError("Invalid video FPS")
        if type(max_pending_samples) is not int or max_pending_samples < 1024:
            raise ValueError("Output capacity must be at least 1024 samples")
        if not math.isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError("Invalid I/O timeout")
        binary = str(sox) if sox else shutil.which("sox")
        if not binary or not Path(binary).is_file():
            raise FileNotFoundError("SoX executable is required")
        self.factor = 25 / fps
        self.command = [binary, "--buffer", "1024", "-D",
                        "-t", "s16", "-r", "16000", "-c", "1", "-L", "-",
                        "-t", "f32", "-r", "16000", "-c", "1", "-L", "-",
                        "tempo", str(self.factor)]
        self.max_pending_samples = max_pending_samples
        self.io_timeout = io_timeout
        self.input_samples = 0
        self.output_samples = 0
        self.peak_pending_samples = 0
        self.finished = False
        self.closed = False
        self._eof = False
        self._pending = bytearray()
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=self._stderr, bufsize=0)
        except BaseException:
            self._stderr.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def _error(self):
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell() - 2048))
        return self._stderr.read().decode(errors="replace")

    def _read(self):
        data = os.read(self.process.stdout.fileno(), 4096)
        if not data:
            self._eof = True
            return
        self._pending.extend(data)
        pending = len(self._pending) // 4
        self.peak_pending_samples = max(self.peak_pending_samples, pending)
        if pending > self.max_pending_samples:
            raise BufferError("Prepared PCM queue capacity exceeded; drain before pushing")

    def read_available(self, *, wait_seconds=0):
        if self.closed:
            raise RuntimeError("PCM preparer is closed")
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("Invalid read wait")
        deadline = time.monotonic() + wait_seconds
        while not self._eof:
            ready, _, _ = select.select([self.process.stdout], [], [],
                                        max(0, deadline - time.monotonic()))
            if not ready:
                break
            self._read()
            # Keep the original deadline: one output chunk can arrive before
            # the other chunks produced from the same input packet. Stopping
            # at that brief gap accumulates pipe backlog and feeds unneeded
            # future packets. Never extend the caller's total wait budget.
        count = len(self._pending) // 4 * 4
        result = bytes(self._pending[:count])
        del self._pending[:count]
        self.output_samples += count // 4
        return result

    def push(self, pcm16):
        if self.closed or self.finished:
            raise RuntimeError("PCM preparer cannot accept more input")
        if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
            raise ValueError("Expected nonempty mono little-endian PCM16 bytes")
        samples = len(pcm16) // 2
        if samples > self.max_packet_samples:
            raise ValueError("Packet exceeds the 200 ms wire limit")
        view = memoryview(pcm16)
        deadline = time.monotonic() + self.io_timeout
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SoX input/backpressure timeout")
            readable, writable, _ = select.select(
                [self.process.stdout] if not self._eof else [],
                [self.process.stdin], [], remaining)
            if readable:
                self._read()
            if writable:
                try:
                    written = os.write(self.process.stdin.fileno(), view)
                except BrokenPipeError as error:
                    raise RuntimeError("SoX input closed: " + self._error()) from error
                view = view[written:]
        self.input_samples += samples

    def finish(self):
        if self.closed or self.finished:
            raise RuntimeError("PCM preparer cannot finish twice")
        self.finished = True
        self.process.stdin.close()
        deadline = time.monotonic() + self.io_timeout
        while not self._eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SoX EOS drain timeout")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if ready:
                self._read()
        code = self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if code:
            raise RuntimeError(f"SoX exited {code}: {self._error()}")
        if len(self._pending) % 4:
            raise ValueError("Incomplete float32 sample at EOS")
        return self.read_available()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self._stderr):
            if not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
