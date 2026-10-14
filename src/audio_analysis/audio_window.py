"""Sample bounds for offline finite-lookahead quality experiments.

Times refer to the original video's frame grid. Tempo-adjusted 16 kHz PCM
has 640 samples per feature, with 25 encoder features per prepared second.
Preparing the full PCM offline is not itself a causal audio front end.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AudioWindow:
    condition_start_frame: int
    condition_end_frame: int
    window_start_frame: int
    window_end_frame: int
    prepared_start_sample: int
    prepared_end_sample: int
    total_prepared_samples: int
    history_seconds: float
    extra_lookahead_seconds: float
    explicit_right_context_frames: int
    fps: int

    def as_dict(self):
        return asdict(self)


def audio_window(condition_start, condition_end, *, total_samples, fps,
                 history_seconds, extra_lookahead_seconds):
    if (not isinstance(fps, int) or fps <= 0 or total_samples < 640 or
            condition_start < 0 or condition_end <= condition_start):
        raise ValueError("Invalid audio grid or conditioning interval")
    if (not math.isfinite(history_seconds) or history_seconds <= 0 or
            not math.isfinite(extra_lookahead_seconds) or extra_lookahead_seconds < 0):
        raise ValueError("History must be positive and lookahead finite and non-negative")
    total_frames = total_samples // 640
    history_frames = math.floor(history_seconds * fps + 1e-7)
    # Ceil never silently grants less future than the declared frame window.
    extra_frames = math.ceil(extra_lookahead_seconds * fps - 1e-7)
    start = max(0, condition_end - history_frames)
    if start > max(0, condition_start - 2):
        raise ValueError("History is too short for conditioning and its left context")
    end = min(total_frames, condition_end + 2 + extra_frames)
    if start >= end:
        raise ValueError("Conditioning window has no available audio")
    # At EOS include the same sub-frame PCM tail as the full-input reference.
    end_sample = total_samples if end == total_frames else end * 640
    return AudioWindow(
        condition_start, condition_end, start, end, start * 640, end_sample,
        total_samples, history_seconds, extra_lookahead_seconds, 2, fps,
    )
