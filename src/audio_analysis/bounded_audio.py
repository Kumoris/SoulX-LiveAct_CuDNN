"""Encode only the bounded PCM window for each offline quality-test block."""

import json
from pathlib import Path
import time

import torch

from src.audio_analysis.audio_window import audio_window
from util_liveact import get_audio_emb


class BoundedAudioConditioner:
    def __init__(self, prepared_pcm, encode, *, fps, history_seconds,
                 extra_lookahead_seconds, device, log_path=None, repro_trace=None):
        self.pcm = prepared_pcm
        self.encode = encode
        self.fps = fps
        self.history_seconds = history_seconds
        self.extra_lookahead_seconds = extra_lookahead_seconds
        self.device = device
        self.log_path = Path(log_path) if log_path else None
        self.repro_trace = repro_trace

    def __call__(self, condition_start, condition_end, *, block):
        window = audio_window(
            condition_start, condition_end, total_samples=self.pcm.numel(),
            fps=self.fps, history_seconds=self.history_seconds,
            extra_lookahead_seconds=self.extra_lookahead_seconds,
        )
        tick = time.perf_counter()
        # No full-utterance embeddings are computed on this path.
        target = (torch.device("cuda", self.device) if isinstance(self.device, int)
                  else torch.device(self.device))
        devices = ([] if target.type != "cuda" else
                   [target.index if target.index is not None else torch.cuda.current_device()])
        # HF 4.49 wav2vec draws LayerDrop randomness even in eval mode.
        # Isolate it so per-block audio encoding cannot perturb diffusion RNG.
        pcm = self.pcm[window.prepared_start_sample:window.prepared_end_sample]
        if self.repro_trace:
            self.repro_trace.tensor('prepared_pcm', pcm, block=block)
        with torch.random.fork_rng(devices=devices):
            encoded = self.encode(pcm)
        if self.repro_trace:
            self.repro_trace.tensor('encoded_audio', encoded, block=block)
        result = get_audio_emb(
            encoded, condition_start - window.window_start_frame,
            condition_end - window.window_start_frame, self.device,
        )
        if self.log_path:
            record = {"block": block, **window.as_dict(),
                      "encoded_frames": encoded.shape[0],
                      "condition_shape": list(result.shape),
                      "encode_and_gather_cpu_wall_seconds": time.perf_counter() - tick,
                      "effective_extra_lookahead_frames":
                          max(0, window.window_end_frame - condition_end - 2),
                      "mode": "offline_fixed_pcm_bounded_encoder_quality_test",
                      "diffusion_rng_isolated": True,
                      "limitation": "Whole PCM was tempo-prepared offline; this is not an accepted live-audio path. CPU timing can include CUDA enqueue rather than completed GPU work."}
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return result
