"""Experimental first-block condition prefix, keeping the model's 53-frame layout."""

import json
from pathlib import Path


FIRST_BLOCK_CONDITION_FRAMES = 21
MODEL_CONDITION_FRAMES = 53


class FirstBlockAudioConditioner:
    """Request only frame 0..20 on block 0; all subsequent requests are unchanged.

    The current six-latent first block selects audio projections 0..5. The
    projection groups are frame 0, then groups of four frames; consequently
    only 21 condition frames feed those six projections. Zero padding keeps
    all projection GEMM shapes fixed. Shorter wav2vec context can change even
    the retained features, so the experiment still requires video review.
    """

    def __init__(self, base, *, log_path=None):
        self.base = base
        self.log_path = Path(log_path) if log_path else None

    def __call__(self, condition_start, condition_end, *, block):
        if condition_end - condition_start != MODEL_CONDITION_FRAMES or block < 0:
            raise ValueError('First-block experiment requires the frozen 53-frame condition layout')
        if block == 0 and condition_start != 0:
            raise ValueError('First block must start at audio frame zero')
        used_end = FIRST_BLOCK_CONDITION_FRAMES if block == 0 else condition_end
        result = self.base(condition_start, used_end, block=block)
        if result.ndim != 5 or result.shape[1] != used_end - condition_start:
            raise ValueError('Unexpected condition prefix shape')
        if block == 0:
            shape = list(result.shape)
            shape[1] = MODEL_CONDITION_FRAMES
            padded = result.new_zeros(shape)
            padded[:, :FIRST_BLOCK_CONDITION_FRAMES] = result
            result = padded
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps({'block': block,
                    'requested_condition_start': condition_start,
                    'requested_condition_end': condition_end,
                    'encoded_condition_end': used_end,
                    'returned_shape': list(result.shape),
                    'unused_condition_padding': 'zeros' if block == 0 else None,
                    'limitation': 'Shorter encoder context is a quality candidate, not an equivalence or latency acceptance.'}) + '\n')
        return result
