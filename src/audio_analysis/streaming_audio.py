"""Incremental PCM preparation replay and bounded wav2vec conditioning.

Replay supplies a local test fixture in 200 ms packets on demand, without
real-time pacing. This verifies preparation/conditioning, not the live web
service or generation throughput.
"""

import json
import math
from pathlib import Path
import time
import weakref

import numpy as np

from src.audio_analysis.audio_window import audio_window
from src.audio_analysis.streaming_pcm import StreamingTempoPCM


class ReplayPreparedPCM:
    def __init__(self, pcm16, *, fps=20, packet_samples=3200, sox=None,
                 max_buffer_samples=128000):
        if not isinstance(pcm16, bytes) or len(pcm16) < 16000 or len(pcm16) % 2:
            raise ValueError("Replay requires at least 0.5 seconds of mono PCM16")
        if type(packet_samples) is not int or not 1 <= packet_samples <= 3200:
            raise ValueError("Replay packet size must be 1–3200 samples")
        self.source = pcm16
        self.fps = fps
        self.packet_samples = packet_samples
        self.max_buffer_samples = max_buffer_samples
        self.preparer = StreamingTempoPCM(fps=fps, sox=sox)
        # Also reclaim the private DSP process on Python error/normal exit.
        self._finalizer = weakref.finalize(self, self.preparer.close)
        self.source_samples = 0
        self.prepared_samples = 0
        self.buffer_start = 0
        self.buffer = np.empty(0, dtype=np.float32)
        self.source_eof = False
        self.peak_buffer_samples = 0
        self.first_output_input_sample = None

    def _append(self, data):
        samples = np.frombuffer(data, dtype="<f4") * np.float32(3)
        if not samples.size:
            return
        if self.first_output_input_sample is None:
            self.first_output_input_sample = self.source_samples
        if self.buffer.size + samples.size > self.max_buffer_samples:
            raise BufferError("Bounded preparation replay buffer exceeded")
        self.buffer = np.concatenate((self.buffer, samples))
        self.prepared_samples += samples.size
        self.peak_buffer_samples = max(self.peak_buffer_samples, self.buffer.size)

    def window(self, condition_start, condition_end, *, history_seconds,
               extra_lookahead_seconds):
        tick = time.perf_counter()
        # Before EOS, reserve one extra feature to avoid granting a partial
        # tail merely because the caller's horizon is the current buffer end.
        horizon = (condition_end + 2 +
                   math.ceil(extra_lookahead_seconds * self.fps - 1e-7)) * 640
        planned = audio_window(
            condition_start, condition_end,
            total_samples=max(horizon + 640, self.prepared_samples),
            fps=self.fps, history_seconds=history_seconds,
            extra_lookahead_seconds=extra_lookahead_seconds)
        if planned.prepared_start_sample < self.buffer_start:
            raise ValueError("Requested history has already been evicted")
        discard = min(self.buffer.size,
                      planned.prepared_start_sample - self.buffer_start)
        self.buffer = self.buffer[discard:].copy()
        self.buffer_start += discard
        while self.prepared_samples < horizon and not self.source_eof:
            start = self.source_samples * 2
            end = min(len(self.source), start + self.packet_samples * 2)
            self.preparer.push(self.source[start:end])
            self.source_samples = end // 2
            self._append(self.preparer.read_available(wait_seconds=0.01))
            if end == len(self.source):
                self.source_eof = True
                self._append(self.preparer.finish())
        total = self.prepared_samples if self.source_eof else max(horizon + 640,
                                                                 self.prepared_samples)
        window = audio_window(
            condition_start, condition_end, total_samples=total,
            fps=self.fps, history_seconds=history_seconds,
            extra_lookahead_seconds=extra_lookahead_seconds)
        start = window.prepared_start_sample - self.buffer_start
        end = window.prepared_end_sample - self.buffer_start
        if start < 0 or end > self.buffer.size:
            raise ValueError("Requested prepared PCM is not available")
        pcm = self.buffer[start:end].copy()
        metadata = {
            "input_samples_received": self.source_samples,
            "input_seconds_received": self.source_samples / 16000,
            "prepared_samples_emitted": self.prepared_samples,
            "source_eof_received": self.source_eof,
            "buffer_start_sample": self.buffer_start,
            "retained_prepared_samples": self.buffer.size,
            "peak_retained_prepared_samples": self.peak_buffer_samples,
            "first_output_input_sample": self.first_output_input_sample,
            "preparation_excess_input_seconds": max(
                0, self.source_samples / 16000 -
                window.prepared_end_sample * (25 / self.fps) / 16000),
            "preparation_and_wait_wall_seconds": time.perf_counter() - tick,
            "packet_samples": self.packet_samples,
            "sox_command": self.preparer.command,
        }
        return window, pcm, metadata

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class StreamingAudioConditioner:
    def __init__(self, replay, encode, *, history_seconds,
                 extra_lookahead_seconds, device, log_path=None, repro_trace=None):
        self.replay = replay
        self.encode = encode
        self.history_seconds = history_seconds
        self.extra_lookahead_seconds = extra_lookahead_seconds
        self.device = device
        self.log_path = Path(log_path) if log_path else None
        self.repro_trace = repro_trace

    def __call__(self, condition_start, condition_end, *, block):
        import torch
        from util_liveact import get_audio_emb
        window, pcm, metadata = self.replay.window(
            condition_start, condition_end, history_seconds=self.history_seconds,
            extra_lookahead_seconds=self.extra_lookahead_seconds)
        if self.repro_trace:
            self.repro_trace.tensor('prepared_pcm', pcm, block=block)
        target = (torch.device("cuda", self.device) if isinstance(self.device, int)
                  else torch.device(self.device))
        devices = ([] if target.type != "cuda" else
                   [target.index if target.index is not None else torch.cuda.current_device()])
        with torch.random.fork_rng(devices=devices):
            encoded = self.encode(torch.from_numpy(pcm))
        if self.repro_trace:
            self.repro_trace.tensor('encoded_audio', encoded, block=block)
        result = get_audio_emb(encoded, condition_start - window.window_start_frame,
                               condition_end - window.window_start_frame, self.device)
        if self.log_path:
            row = {"block": block, **window.as_dict(), **metadata,
                   "encoded_frames": encoded.shape[0],
                   "condition_shape": list(result.shape),
                   "diffusion_rng_isolated": True,
                   "mode": "packet_replay_stateful_tempo_bounded_encoder",
                   "limitation": "Test fixture replay is not wall-clock paced; DSP consumes additional input reported separately. Live browser input and >=20 FPS are not accepted."}
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return result
