"""Frame accounting shared by offline and streaming generation."""
import math

def target_frame_count(seconds, fps):
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("duration and FPS must be positive and finite")
    return max(1, math.ceil(seconds * fps - 1e-7))

def required_blocks(frames, latent_blocks=(6, 8), temporal_stride=4):
    if frames <= 0 or len(latent_blocks) != 2 or min(latent_blocks) <= 0 or temporal_stride <= 0:
        raise ValueError("invalid block dimensions")
    first = (latent_blocks[0] - 1) * temporal_stride + 1
    following = latent_blocks[1] * temporal_stride
    return 1 + math.ceil(max(0, frames - first) / following)
