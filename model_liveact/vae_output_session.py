"""Experimentally replace repeated output concatenation, preserving VAE state."""
import ast
from functools import wraps
import inspect
import textwrap

import torch


class VAEOutputSession:
    def __init__(self, module):
        self.cls = module.WanVAE_
        self.original = self.cls.decode
        self.installed = False
        self.drop_frames = 0
        self.calls = 0
        self.discarded_frames = 0

    def _assemble(self, parts):
        remaining = self.drop_frames
        kept = []
        for part in parts:
            skip = min(remaining, part.shape[2])
            remaining -= skip
            if skip < part.shape[2]:
                kept.append(part[:, :, skip:] if skip else part)
        if remaining or not kept:
            raise ValueError('Warm-up trim must leave at least one decoded frame')
        self.calls += 1
        self.discarded_frames += self.drop_frames
        return torch.cat(kept, dim=2)

    def install(self):
        if self.installed:
            raise ValueError('VAE output session already installed')
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.original)))
        stamp = lambda node: ast.dump(node, include_attributes=False)
        expected = ast.parse('''
for i in range(iter_):
    self._conv_idx = [0]
    if i == 0:
        out = self.decoder(x[:, :, i:i+1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
    else:
        out_ = self.decoder(x[:, :, i:i+1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        out = torch.cat([out, out_], 2)
''').body[0]
        body = tree.body[0].body
        matches = [i for i, node in enumerate(body) if stamp(node) == stamp(expected)]
        if len(matches) != 1:
            raise ValueError('Exactly one original temporal VAE decode loop required')
        index = matches[0]
        replacement = ast.parse('''
_r2_parts = []
for i in range(iter_):
    self._conv_idx = [0]
    out_ = self.decoder(x[:, :, i:i+1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
    _r2_parts.append(out_)
out = _r2_assemble(_r2_parts)
''').body
        body[index:index+1] = replacement
        tree = ast.fix_missing_locations(tree)
        namespace = dict(self.original.__globals__)
        namespace['_r2_assemble'] = self._assemble
        exec(compile(tree, inspect.getsourcefile(self.original), 'exec'), namespace)
        self.cls.decode = wraps(self.original)(namespace['decode'])
        self.installed = True

    def decode(self, vae, latent, *, drop_frames=0):
        if not self.installed or self.drop_frames:
            raise ValueError('Inactive or reentrant VAE output session')
        if drop_frames not in (0, 9):
            raise ValueError('Only the existing nine-frame warm-up trim is supported')
        self.drop_frames = drop_frames
        try:
            return vae.decode(latent)
        finally:
            self.drop_frames = 0

    def report(self):
        return dict(calls=self.calls, discarded_rgb_frames=self.discarded_frames,
                    weight_math_changed=False, temporal_state_changed=False,
                    warmup_decoder_calls_skipped=False, distributed_split_changed=False)

    def close(self):
        if self.installed:
            self.cls.decode = self.original
        self.installed = False
        self.drop_frames = 0
