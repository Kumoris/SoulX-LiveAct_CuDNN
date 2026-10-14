"""Opt-in SP4 adaptation preserving the accepted SP2 virtual history layout.

SP8 is deliberately rejected until uneven sequence shards have a masked transport.
No weights, compression operator, block size, or temporal conditioning is changed.
"""
import ast
from collections import OrderedDict
from functools import wraps
import inspect
import math
import textwrap

import torch
import torch.distributed as dist
from torch.nn import functional as F

from model_liveact.positional_layout import RopeLayoutSession


class ParallelGeometry:
    def __init__(self, frame_tokens, world, rank):
        if world not in (2, 4) or not 0 <= rank < world:
            raise ValueError('Only validated equal-shard SP2/SP4 are supported')
        if frame_tokens <= 0 or any(frames * frame_tokens % world for frames in (6, 8, 14)):
            raise ValueError('Uneven token shards require a separate masked implementation')
        self.frame_tokens, self.world, self.rank = frame_tokens, world, rank
        self.history = 6 * frame_tokens // world
        self.current = 8 * frame_tokens // world
        self.virtual_size = world // 2
        self.virtual_rank = rank // self.virtual_size
        self.member = rank % self.virtual_size

    def key_ids(self, first, device):
        history = torch.arange(self.rank*self.history, (self.rank+1)*self.history, device=device)
        if first:
            return history
        current = torch.arange(6*self.frame_tokens+self.rank*self.current,
                               6*self.frame_tokens+(self.rank+1)*self.current, device=device)
        return torch.cat((history, current))

    def reconstruct_virtual(self, parts):
        if len(parts) != self.virtual_size or any(p.shape[1] != self.history+self.current for p in parts):
            raise ValueError('Incomplete virtual SP2 cache gather')
        return torch.cat([p[:, :self.history] for p in parts] +
                         [p[:, self.history:] for p in parts], dim=1)

    def history_slice(self, virtual_cache):
        return virtual_cache[:, self.member*self.history:(self.member+1)*self.history]

    def audio_segments(self, frames):
        count = frames*self.frame_tokens//self.world
        offset, stop = self.rank*count, (self.rank+1)*count
        segments = []
        position = offset
        while position < stop:
            frame = position//self.frame_tokens
            end = min(stop, (frame+1)*self.frame_tokens)
            segments.append((position-offset, end-offset, frame))
            position = end
        return segments


def actual_compression(forward):
    """Execute the original compression branch, including its Conv1d semantics."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(forward)))
    branch = next(n for n in tree.body[0].body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'update_cache')
    wrapper = ast.parse('def compress(self, k_cache, v_cache, kv_cache, sp_rank, frame_seqlen):\n    pass').body[0]
    wrapper.body = branch.body
    ns = dict(forward.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 inspect.getsourcefile(forward), 'exec'), ns)
    return ns['compress']


class ParallelGeometrySession:
    def __init__(self, module, *, gather=None, fused_rotary=None, query_rotary=None):
        self.module, self.gather, self.fused_rotary = module, gather, fused_rotary
        self.original_forward = module.WanSelfAttention.forward
        self.query_rotary = query_rotary
        self.compress = actual_compression(self.original_forward)
        self.groups, self.restores, self.tables = {}, [], OrderedDict()
        self.frequency = None
        self.indices = {}
        self.counts = dict(cache_calls=0, compression_calls=0, subgroup_gathers=0,
                           audio_calls=0, partial_audio_segments=0, key_rotary_calls=0)

    def initialize_groups(self, world):
        if self.gather is not None or world == 2 or self.groups:
            return
        if (not dist.is_initialized() or dist.get_world_size() != world
                or self.module.get_sequence_parallel_world_size() != world):
            raise ValueError('Pilot requires one whole-world sequence-parallel group')
        # Every rank creates both groups in the same order, before inference.
        for virtual in range(2):
            self.groups[virtual] = dist.new_group(ranks=list(range(virtual*2, virtual*2+2)))

    def collect(self, tensor, geometry):
        if self.gather is not None:
            return self.gather(tensor, geometry)
        self.initialize_groups(geometry.world)
        outputs = [torch.empty_like(tensor) for _ in range(geometry.virtual_size)]
        # Main thread only; previous attention collectives have completed.
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
        dist.all_gather(outputs, tensor.contiguous(), group=self.groups[geometry.virtual_rank])
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
        self.counts['subgroup_gathers'] += 1
        return outputs

    def prepare_cache(self, instance, k_cache, v_cache, k, v, kv_cache,
                      grid_sizes, sp_size, sp_rank, start_idx, end_idx, update_cache):
        frame = math.prod(grid_sizes[0][1:]).item()
        g = ParallelGeometry(frame, sp_size, sp_rank)
        first = start_idx == 0
        if (end_idx != (6 if first else 14)*frame
                or k_cache.shape[1] != g.history+g.current
                or k.shape[1] != (g.history if first else g.current)
                or kv_cache.get('offload_cache') or kv_cache.get('fp8_kv_cache')):
            raise ValueError('Unsupported cache/block/precision configuration')
        if update_cache:
            if first:
                raise ValueError('First block cannot compress prior history')
            if sp_size == 2:
                self.compress(instance, k_cache, v_cache, kv_cache, sp_rank, frame)
            else:
                # Gather K/V together, reconstruct the exact original seven-frame
                # virtual cache, apply the original branch, and reshard history.
                joined = torch.cat((k_cache, v_cache), dim=0)
                parts = self.collect(joined, g)
                virtual = g.reconstruct_virtual(parts)
                vk, vv = virtual.chunk(2, dim=0)
                self.compress(instance, vk, vv, kv_cache, g.virtual_rank, frame)
                k_cache[:, :g.history].copy_(g.history_slice(vk))
                v_cache[:, :g.history].copy_(g.history_slice(vv))
            self.counts['compression_calls'] += 1
        destination = slice(0, g.history) if first else slice(g.history, None)
        k_cache[:, destination] = k
        v_cache[:, destination] = v
        self.counts['cache_calls'] += 1
        length = g.history if first else g.history+g.current
        identity = (frame, sp_size, sp_rank, first, k.device)
        if identity not in self.indices:
            self.indices[identity] = g.key_ids(first, k.device)
        return k_cache[:, :length], v_cache[:, :length], (identity, self.indices[identity])

    def key_rope(self, x, grid_sizes, freqs, key_ids):
        geometry_key, ids = key_ids
        _, h, w = grid_sizes.tolist()[0]
        half = x.shape[-1]//2
        try:
            stamp = (id(freqs), freqs._version, freqs.device, freqs.dtype)
        except RuntimeError:
            stamp = None
        if stamp is None or self.frequency is None or self.frequency[0] != stamp:
            self.tables.clear()
            self.frequency = (stamp, freqs)
        # Geometry, not activation values, defines this immutable table.
        identity = (geometry_key, x.shape[1], h, w, half, x.device)
        if identity not in self.tables:
            full = RopeLayoutSession._make_table(freqs, half, 14, h, w, 0, x.device)
            self.tables[identity] = full.index_select(0, ids).contiguous()
            if len(self.tables) > 8:
                self.tables.popitem(last=False)
        table = self.tables[identity]
        self.counts['key_rotary_calls'] += 1
        if self.fused_rotary is not None and x.device.type == 'cuda' and x.dtype == torch.bfloat16:
            return self.fused_rotary(x, table)
        values = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], half, 2))
        return torch.view_as_real(values*table.unsqueeze(0)).flatten(-2)

    def audio(self, instance, x, encoder_hidden_states, sp_size, sp_rank, shape=None, start_f=0):
        frames, h, w = [int(value) for value in shape]
        g = ParallelGeometry(h*w, sp_size, sp_rank)
        if x.shape[0] != 1 or x.shape[1] != frames*h*w//sp_size:
            raise ValueError('Pilot audio requires one equal token shard')
        states = encoder_hidden_states.squeeze(0)
        q = instance.q_linear(x).view(1, x.shape[1], instance.num_heads,
                                     instance.head_dim).permute(0, 2, 1, 3)
        if instance.qk_norm:
            q = instance.q_norm(q)
        kv = instance.kv_linear(states).view(states.shape[0], states.shape[1], 2,
                                             instance.num_heads, instance.head_dim)
        outputs = []
        for begin, end, frame in g.audio_segments(frames):
            index = start_f+frame
            if index >= kv.shape[0]:
                raise ValueError('Audio frame index exceeds supplied conditions')
            key, value = kv[index:index+1].permute(2, 0, 3, 1, 4).unbind(0)
            if instance.qk_norm:
                key = instance.add_k_norm(key)
            result = F.scaled_dot_product_attention(q[:, :, begin:end], key, value,
                                                     dropout_p=0.0, is_causal=False)
            outputs.append(result.transpose(1, 2).reshape(1, end-begin, instance.dim))
            if end-begin != h*w:
                self.counts['partial_audio_segments'] += 1
        self.counts['audio_calls'] += 1
        return instance.proj_drop(instance.proj(torch.cat(outputs, dim=1)))

    def transformed_forward(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.original_forward)))
        body = tree.body[0].body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'update_cache')
        stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'f_list')
        expected = ['update_cache', 'start_idx != 0', 'kv_idx', 'rope_list', 'f_list']
        actual = [ast.unparse(n.test) if isinstance(n, ast.If) else ast.unparse(n.targets[0])
                  for n in body[start:stop+1]]
        if actual != expected:
            raise ValueError('Unknown original KV/rotary source; refusing adapter')
        replacement = ast.parse('key, value, key_ids = _parallel_prepare(self, k_cache, v_cache, k, v, kv_cache, grid_sizes, sp_size, sp_rank, start_idx, end_idx, update_cache)').body
        body[start:stop+1] = replacement
        counts = dict(key=0, value=0)
        class Replace(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'rope_apply':
                    counts['key'] += 1
                    return ast.parse('_parallel_rope(key, grid_sizes, freqs, key_ids)', mode='eval').body
                return self.generic_visit(node)
            def visit_Subscript(self, node):
                if ast.unparse(node) == 'v_cache[:, kv_idx]':
                    counts['value'] += 1
                    return ast.Name(id='value', ctx=ast.Load())
                return self.generic_visit(node)
        tree = ast.fix_missing_locations(Replace().visit(tree))
        if counts != dict(key=1, value=1):
            raise ValueError('Unknown original attention inputs')
        ns = dict(self.original_forward.__globals__)
        ns.update(_parallel_prepare=self.prepare_cache, _parallel_rope=self.key_rope)
        if self.query_rotary is not None:
            ns['causal_rope_apply'] = self.query_rotary
        exec(compile(tree, inspect.getsourcefile(self.original_forward), 'exec'), ns)
        return wraps(self.original_forward)(ns['forward'])

    def install(self):
        cls = self.module.WanSelfAttention
        forward = self.transformed_forward()
        original_init = cls.init_kvidx
        session = self
        @wraps(original_init)
        def initialize(instance, frame_len, world_size):
            rank = session.module.get_sequence_parallel_rank()
            ParallelGeometry(frame_len, world_size, rank)
            session.initialize_groups(world_size)
            original_init(instance, frame_len, world_size)
        cls.forward, cls.init_kvidx = forward, initialize
        self.restores += [(cls, 'forward', self.original_forward), (cls, 'init_kvidx', original_init)]
        audio = self.module.SingleStreamAttention
        original_audio = audio.forward
        def audio_forward(instance, *args, **kwargs):
            return session.audio(instance, *args, **kwargs)
        audio.forward = audio_forward
        self.restores.append((audio, 'forward', original_audio))

    def clear(self):
        self.tables.clear()
        self.indices.clear()
        self.frequency = None

    def report(self):
        return dict(**self.counts, supported_world_sizes=[2, 4], sp8_supported=False,
                    temporal_compression_changed=False, weights_changed=False,
                    resident_tables=len(self.tables), subgroup_count=len(self.groups))

    def close(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.restores.clear()
        self.clear()
