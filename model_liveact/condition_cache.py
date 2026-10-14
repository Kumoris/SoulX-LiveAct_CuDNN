"""Opt-in, task-bounded condition projection reuse; model weights stay intact.

Only the experimental entry point installs these instance forward wrappers.
Self attention, diffusion steps, queries and VAE are never cached here.
"""

from contextvars import ContextVar
from functools import wraps
import inspect
import weakref

import torch
from torch import nn


_SCOPE = ContextVar('liveact_condition_projection_scope', default=None)


def tensor_stamp(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors have no mutation counter; do not reuse them.
        return None
    return (id(tensor), version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), str(tensor.device), tensor.dtype)


def parameter_stamps(module):
    parameters = tuple(module.parameters())
    stamps = tuple(tensor_stamp(p) for p in parameters)
    return stamps, parameters, all(s is not None for s in stamps)


def execution_stamp():
    return (torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu'),
            torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
            torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def argument_metadata(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                str(value.device), value.dtype)
    if isinstance(value, (list, tuple)):
        return tuple(argument_metadata(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, argument_metadata(v)) for k, v in sorted(value.items()))
    return value


class ProjectionEntry:
    """One current value per operation; strong references prevent ID reuse."""

    def __init__(self, name, module, channel, owner, dependencies=()):
        self.name, self.module, self.channel = name, module, channel
        self.owner = weakref.ref(owner)
        self.dependencies = dependencies
        self.original = module.forward
        self.key = self.value = None
        self.references = ()
        self.hits = self.misses = self.bypasses = 0

    def clear(self):
        self.key = self.value = None
        self.references = ()

    def __call__(self, *args, **kwargs):
        scope = _SCOPE.get()
        if (scope is None or scope['owner'] is not self.owner()
                or not scope['enabled'] or self.module.training or torch.is_grad_enabled()):
            self.bypasses += 1
            self.clear()
            return self.original(*args, **kwargs)
        params, references, valid = parameter_stamps(self.module)
        for dependency in self.dependencies:
            extra, refs, okay = parameter_stamps(dependency)
            params += extra
            references += refs
            valid = valid and okay
        if not valid:
            self.bypasses += 1
            self.clear()
            return self.original(*args, **kwargs)
        key = (scope[self.channel], params, scope['execution'],
               argument_metadata(args), argument_metadata(kwargs))
        if self.key == key:
            self.hits += 1
            return self.value
        self.misses += 1
        self.value = self.original(*args, **kwargs)
        self.key = key
        self.references = scope['references'] + references
        return self.value

    def report(self):
        tensors = self.value if isinstance(self.value, (list, tuple)) else (self.value,)
        return {'operation': self.name, 'channel': self.channel,
                'hits': self.hits, 'misses': self.misses, 'bypasses': self.bypasses,
                'resident_entries': int(self.key is not None),
                'cached_tensor_bytes': sum(t.numel() * t.element_size()
                                           for t in tensors if isinstance(t, torch.Tensor))}


class ConditionCacheSession:
    def __init__(self, mode='text'):
        if mode not in ('text', 'text-audio'):
            raise ValueError('Cache mode must be text or text-audio')
        self.mode = mode
        self.entries = []
        self.models = weakref.WeakSet()
        self.task_resets = 0

    def clear(self):
        for entry in self.entries:
            entry.clear()
        self.task_resets += 1

    def install(self, model):
        if model in self.models:
            raise ValueError('Condition cache already installed on this model')
        self.models.add(model)

        def wrap(name, module, channel, dependencies=()):
            if isinstance(module, nn.Identity):
                return
            entry = ProjectionEntry(name, module, channel, model, dependencies)
            self.entries.append(entry)
            original = module.forward

            @wraps(original)
            def forward(*args, **kwargs):
                return entry(*args, **kwargs)
            module.forward = forward

        wrap('text_embedding', model.text_embedding, 'text')
        wrap('img_emb', model.img_emb, 'image')
        for i, block in enumerate(model.blocks):
            for name in ('k', 'v'):
                wrap(f'blocks.{i}.cross_attn.{name}', getattr(block.cross_attn, name), 'text')
            wrap(f'blocks.{i}.cross_attn.norm_k', block.cross_attn.norm_k, 'text',
                 (block.cross_attn.k,))
            for name in ('k_img', 'v_img'):
                wrap(f'blocks.{i}.cross_attn.{name}', getattr(block.cross_attn, name), 'image')
            wrap(f'blocks.{i}.cross_attn.norm_k_img', block.cross_attn.norm_k_img, 'image',
                 (block.cross_attn.k_img,))
            if self.mode == 'text-audio':
                wrap(f'blocks.{i}.audio_cross_attn.kv_linear', block.audio_cross_attn.kv_linear, 'audio')
        if self.mode == 'text-audio':
            wrap('audio_proj', model.audio_proj, 'audio')

        original = model.forward
        signature = inspect.signature(original)

        @wraps(original)
        def forward(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            data = bound.arguments
            context = data.get('context')
            context = tuple(context) if isinstance(context, (list, tuple)) else ()
            clip, audio = data.get('clip_fea'), data.get('audio')
            references = context + tuple(t for t in (clip, audio) if isinstance(t, torch.Tensor))
            stamps = tuple(tensor_stamp(t) for t in references)
            text_weights, _, text_valid = parameter_stamps(model.text_embedding)
            image_weights, _, image_valid = parameter_stamps(model.img_emb)
            audio_weights, _, audio_valid = parameter_stamps(model.audio_proj)
            enabled = (bool(context) and isinstance(clip, torch.Tensor)
                       and isinstance(audio, torch.Tensor) and not model.training
                       and not torch.is_grad_enabled() and all(s is not None for s in stamps)
                       and text_valid and image_valid and audio_valid)
            scope = {'owner': model, 'enabled': enabled, 'references': references,
                     'execution': execution_stamp(),
                     'text': (tuple(tensor_stamp(t) for t in context), text_weights, image_weights),
                     'image': (tensor_stamp(clip), image_weights),
                     'audio': (tensor_stamp(audio), audio_weights,
                               getattr(model, 'audio_window', None), getattr(model, 'vae_scale', None))}
            token = _SCOPE.set(scope)
            try:
                return original(*args, **kwargs)
            finally:
                _SCOPE.reset(token)
        model.forward = forward
        return model

    def report(self):
        rows = [entry.report() for entry in self.entries]
        return {'mode': self.mode, 'task_resets': self.task_resets, 'operations': rows,
                'hits': sum(r['hits'] for r in rows), 'misses': sum(r['misses'] for r in rows),
                'bypasses': sum(r['bypasses'] for r in rows),
                'resident_entries': sum(r['resident_entries'] for r in rows),
                'cached_tensor_bytes': sum(r['cached_tensor_bytes'] for r in rows),
                'cache_scope': 'One current value per operation; cleared at condition_start',
                'weights_changed': False}
