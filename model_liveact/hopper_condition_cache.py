"""Conservative projection reuse that also tracks FP8 weights stored as buffers."""
from model_liveact import condition_cache as base


def parameter_and_buffer_stamps(module):
    # FP8Linear deliberately discards its original Parameter after quantization.
    # Registered FP8 weights/scales remain mutable; tracking Parameters alone
    # would silently reuse stale projections after buffer updates/invalidation.
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    stamps = tuple(base.tensor_stamp(tensor) for tensor in tensors)
    # Track missing/materialized buffer names too. Strong tensor references in
    # ProjectionEntry prevent Python ID reuse. Unversioned inference tensors
    # retain the original conservative bypass rather than assuming immutability.
    structure = tuple((id(child), tuple((name, value is not None)
                       for name, value in child._buffers.items())) for child in module.modules())
    return (structure, stamps), tensors, all(stamp is not None for stamp in stamps)


class HopperConditionCache:
    def __init__(self, mode='text-audio'):
        self.session = base.ConditionCacheSession(mode)
        self.original_stamps = base.parameter_stamps
        self.restores = []
        self.installed = False

    def install_tracker(self):
        if self.installed:
            raise ValueError('Projection buffer tracker already installed')
        base.parameter_stamps = parameter_and_buffer_stamps
        self.installed = True

    def attach(self, model):
        if not self.installed:
            raise ValueError('Install buffer tracker before wrapping actual FP8 modules')
        # Called AFTER enable_fp8_gemm, so the wrapped modules are the actual
        # FP8Linear instances rather than the replaced original nn.Linear.
        originals = [(module, module.forward) for module in model.modules()]
        self.session.install(model)
        self.restores.extend(originals)
        return model

    def clear(self):
        self.session.clear()

    def report(self):
        return dict(**self.session.report(),fp8_registered_buffers_tracked=True,
                    unversioned_inference_tensors_bypass=True,
                    installed_after_fp8_replacement=True,self_attention_cached=False,
                    diffusion_steps_changed=False,real_time_accepted=False)

    def close(self):
        self.clear()
        for module, original in reversed(self.restores):
            module.forward = original
        self.restores.clear()
        if self.installed:
            base.parameter_stamps = self.original_stamps
        self.installed = False
