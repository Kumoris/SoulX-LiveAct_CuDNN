"""Opt-in reuse of immutable RoPE tables and canonical contiguous KV prefixes.

The original FP64 complex multiply and BF16 rounding remain unchanged. This
experiment never caches activations or changes attention, weights, or KV updates.
"""
import ast
from collections import OrderedDict
from functools import wraps
import inspect
import textwrap

import torch


class RopeLayoutSession:
    def __init__(self, module, mode, *, max_tables=8):
        if mode not in ('tables', 'prefix-views'):
            raise ValueError('Unsupported positional layout mode')
        self.module, self.mode, self.max_tables = module, mode, max_tables
        self.original_causal, self.original_rope = module.causal_rope_apply, module.rope_apply
        self.tables = OrderedDict()
        self.frequency = None
        self.indices = {}
        self.restores = []
        self.counts = dict(causal_calls=0, key_calls=0, table_hits=0, table_misses=0,
                           table_bypasses=0, frequency_invalidations=0, task_resets=0,
                           prefix_calls=0, prefix_hits=0, prefix_fallbacks=0)

    def clear(self):
        self.tables.clear()
        self.frequency = None
        self.counts['task_resets'] += 1

    def _stamp(self, freqs):
        if torch.is_grad_enabled() or freqs.requires_grad:
            return None
        try:
            return (id(freqs), freqs._version, tuple(freqs.shape), tuple(freqs.stride()),
                    freqs.storage_offset(), freqs.device, freqs.dtype)
        except RuntimeError:
            return None

    def _table(self, freqs, key, build):
        stamp = self._stamp(freqs)
        if stamp is None:
            self.counts['table_bypasses'] += 1
            return None
        if self.frequency is None or self.frequency[0] != stamp:
            if self.frequency is not None:
                self.counts['frequency_invalidations'] += 1
            self.tables.clear()
            self.frequency = (stamp, freqs)  # Strong reference prevents identity reuse.
        if key in self.tables:
            self.counts['table_hits'] += 1
            self.tables.move_to_end(key)
            return self.tables[key]
        self.counts['table_misses'] += 1
        table = build().clone()  # Own the slice storage; resident bytes stay bounded.
        self.tables[key] = table
        while len(self.tables) > self.max_tables:
            self.tables.popitem(last=False)
        return table

    @staticmethod
    def _make_table(freqs, c, f, h, w, start, device):
        parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        return torch.cat([
            parts[0][start:start+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f*h*w, 1, -1).to(device=device)

    def causal(self, x, grid_sizes, freqs, sp_size, sp_rank, start_frame=0, _f=None):
        self.counts['causal_calls'] += 1
        if self._stamp(freqs) is None:
            self.counts['table_bypasses'] += 1
            return self.original_causal(x, grid_sizes, freqs, sp_size, sp_rank, start_frame, _f)
        s, n, c = x.size(1), x.size(2), x.size(3)//2
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            f = _f if _f else f
            key = ('query', c, f, h, w, s, sp_size, sp_rank, start_frame, x.device)
            table = self._table(freqs, key, lambda: self._make_table(
                freqs, c, f, h, w, start_frame, x.device)[sp_rank*s:(sp_rank+1)*s])
            value = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
            output.append(torch.view_as_real(value*table).flatten(2))
        return torch.stack(output)

    def rope(self, x, grid_sizes, freqs, f_list=(), rope_list=()):
        self.counts['key_calls'] += 1
        if self._stamp(freqs) is None:
            self.counts['table_bypasses'] += 1
            return self.original_rope(x, grid_sizes, freqs, f_list, rope_list)
        n, c = x.size(2), x.size(3)//2
        _, h, w = grid_sizes.tolist()[0]
        output = []
        # Cache the joined table, while retaining the original per-segment math.
        pairs = tuple((tuple(f), tuple(r)) for f, r in zip(f_list, rope_list))
        if any(end-start != r_end-r_start for (start,end),(r_start,r_end) in pairs):
            return self.original_rope(x, grid_sizes, freqs, f_list, rope_list)
        key = ('key', c, h, w, pairs, x.device)
        table = self._table(freqs, key, lambda: torch.cat([
            self._make_table(freqs, c, end-start, h, w, r_start, x.device)
            for (start, end), (r_start, r_end) in pairs], dim=0))
        offset = 0
        for (start, end), _ in pairs:
            length = (end-start)*h*w
            value = torch.view_as_complex(x[0, start*h*w:end*h*w].to(torch.float64)
                                          .reshape(length, n, -1, 2))
            output.append(torch.view_as_real(value*table[offset:offset+length]).flatten(2))
            offset += length
        return torch.cat(output, dim=0).unsqueeze(0)

    def register_indices(self, indices):
        """Validate the arange once at initialization, outside measured blocks."""
        expected = torch.arange(indices.numel(), dtype=indices.dtype, device=indices.device)
        if indices.ndim != 1 or not torch.equal(indices, expected):
            raise ValueError('KV prefix optimization requires a canonical contiguous arange')
        self.indices[id(indices)] = (indices, indices._version, indices.numel())

    def prefix(self, tensor, indices):
        self.counts['prefix_calls'] += 1
        entry = self.indices.get(id(indices))
        try:
            eligible = (not torch.is_grad_enabled() and entry is not None
                        and entry[0] is indices and entry[1] == indices._version
                        and tensor.ndim == 4 and tensor.shape[1] >= entry[2])
        except RuntimeError:
            eligible = False
        if not eligible:
            self.counts['prefix_fallbacks'] += 1
            return tensor[:, indices]
        self.counts['prefix_hits'] += 1
        return tensor[:, :entry[2]]

    def _prefix_forward(self, original):
        # Transform only the two gathers in the actual, frozen source. Retain
        # compression, KV writes, collectives, attention, and projection verbatim.
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        found = {'k_cache': 0, 'v_cache': 0}

        class ReplaceGathers(ast.NodeTransformer):
            def visit_Subscript(self, node):
                node = self.generic_visit(node)
                if (isinstance(node.value, ast.Name) and node.value.id in found
                        and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                        and isinstance(node.slice.elts[0], ast.Slice)
                        and all(getattr(node.slice.elts[0], k) is None for k in ('lower','upper','step'))
                        and isinstance(node.slice.elts[1], ast.Name)
                        and node.slice.elts[1].id == 'kv_idx'):
                    found[node.value.id] += 1
                    return ast.copy_location(ast.Call(func=ast.Name(id='_r2_prefix',ctx=ast.Load()),
                        args=[node.value,node.slice.elts[1]],keywords=[]),node)
                return node

        tree = ast.fix_missing_locations(ReplaceGathers().visit(tree))
        if found != {'k_cache': 1, 'v_cache': 1}:
            raise ValueError('Expected exactly the original key and value prefix gathers')
        namespace = dict(original.__globals__)
        namespace.update(_r2_prefix=self.prefix, causal_rope_apply=self.causal, rope_apply=self.rope)
        exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
        return wraps(original)(namespace[original.__name__])

    def install(self):
        self.module.causal_rope_apply, self.module.rope_apply = self.causal, self.rope
        self.restores += [(self.module,'causal_rope_apply',self.original_causal),
                          (self.module,'rope_apply',self.original_rope)]
        if self.mode == 'prefix-views':
            cls = self.module.WanSelfAttention
            original_init, original_forward = cls.init_kvidx, cls.forward
            session = self

            @wraps(original_init)
            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                session.register_indices(instance.kv_idx0)
                session.register_indices(instance.kv_idx2)

            patched = self._prefix_forward(original_forward)
            cls.init_kvidx, cls.forward = initialize, patched
            self.restores += [(cls,'init_kvidx',original_init),(cls,'forward',original_forward)]

    def report(self):
        return {'mode':self.mode, **self.counts, 'resident_tables':len(self.tables),
                'resident_table_bytes':sum(t.numel()*t.element_size() for t in self.tables.values()),
                'registered_prefix_indices':len(self.indices),
                'rotary_compute_dtype':'float64/complex128', 'activation_cache':False,
                'weights_changed':False,'attention_backend_changed':False}

    def close(self):
        for obj, name, value in reversed(self.restores):
            setattr(obj,name,value)
        self.restores.clear()
        self.tables.clear()
        self.frequency = None
        self.indices.clear()
