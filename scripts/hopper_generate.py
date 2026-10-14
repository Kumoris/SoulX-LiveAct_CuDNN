"""Dual-Hopper 20 FPS candidate; accepted bounded audio and visible wave remain.

FP8 GEMM is optional and must pass its own video quality/performance review.
No A800 or single-GPU fallback is permitted in this migration entry.
"""
import json
import os
from pathlib import Path


def main():
    from scripts.hopper_runtime_cache import configure_rank_cache, pointwise_cache_smoke

    if int(os.environ.get('WORLD_SIZE', '0')) != 2:
        raise ValueError('Use two actual H100/H200 ranks')
    runtime_cache = configure_rank_cache(Path.cwd())
    import torch
    from scripts.hopper_preflight import validate_device

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    prop = torch.cuda.get_device_properties(local_rank)
    validate_device(prop.name, torch.cuda.get_device_capability(local_rank), prop.total_memory)
    cache_smoke = pointwise_cache_smoke(local_rank, Path.cwd())
    import generate as generator
    import model_liveact.model_memory_sp as model
    import model_liveact.attention as attention
    from lightx2v.utils.envs import GET_USE_CHANNELS_LAST_3D
    from src.audio_analysis.first_block_audio import FirstBlockAudioConditioner
    from model_liveact.rotary_fused_exact import ExactFusedRopeSession
    from model_liveact.pointwise_session import PointwiseSession

    args = generator._parse_args()
    if (args.fps != 20 or args.size != '416*720' or not args.audio_stream_pcm
            or args.audio_lookahead != 0.5 or args.audio_history != 4):
        raise ValueError('Keep 20 FPS, 416x720 and accepted bounded audio settings')
    if args.fp4_gemm or args.fp8_kv_cache or args.offload_cache or args.block_offload:
        raise ValueError('Only FP8 GEMM may differ in this first Hopper comparison')
    if not GET_USE_CHANNELS_LAST_3D():
        raise ValueError('Accepted VAE layout requires USE_CHANNELS_LAST_3D=1')
    # Enable other approximate backends only in separately frozen comparisons.
    model.USE_SAGEATTN = False
    attention.USE_SAGEATTN = False
    original_conditioner = generator.StreamingAudioConditioner
    generator.StreamingAudioConditioner = lambda *a, **kw: FirstBlockAudioConditioner(
        original_conditioner(*a, **kw),
        log_path=Path.cwd() / f'first-block-audio-rank-{os.environ["RANK"]}.jsonl')
    rope = ExactFusedRopeSession(model)
    pointwise = PointwiseSession(model)
    original_trace = generator._p7_trace
    original_enable_fp8 = generator.enable_fp8_gemm
    fp8_hooks = []
    fp8_executed_modules = set()
    fp8_audit = dict(enabled=False, materialized_cutlass_modules=0,
                     cuda_fp8_forward_calls=0, fallback_forward_calls=0,
                     wrapped_cutlass_modules=0, weight_check_stage='after_actual_forward')
    if args.fp8_gemm and os.environ.get('LIVEACT_HOPPER_FP8_AUDIT') == '1':
        from fp8_gemm import FP8Linear
        fp8_audit['enabled'] = True
        def audited_enable(*a, **kw):
            result = original_enable_fp8(*a, **kw)
            modules = [m for m in result.modules() if isinstance(m, FP8Linear)]
            if not modules or any(m._fp8_linear_op.preferred_backend != 'cutlass'
                                  for m in modules):
                raise ValueError('Hopper candidate requires CUTLASS FP8 modules')
            # generate wraps the CPU model before moving it to CUDA. Weights
            # are intentionally quantized lazily on the first CUDA forward.
            # Check the executed path after that forward, rather than rejecting
            # the legitimate, not-yet-materialized CPU wrappers.
            fp8_audit['wrapped_cutlass_modules'] = len(modules)
            def forward_hook(module, inputs, output):
                if (inputs[0].is_cuda and module._fp8_weight is not None
                        and module._fp8_weight.is_cuda
                        and module._fp8_weight.dtype == torch.float8_e4m3fn
                        and module._fp8_linear_op.preferred_backend == 'cutlass'):
                    fp8_audit['cuda_fp8_forward_calls'] += 1
                    fp8_executed_modules.add(id(module))
                    fp8_audit['materialized_cutlass_modules'] = len(fp8_executed_modules)
                else:
                    fp8_audit['fallback_forward_calls'] += 1
                    raise ValueError('Hopper FP8 candidate executed a non-CUTLASS or fallback path')
            fp8_hooks.extend(m.register_forward_hook(forward_hook) for m in modules)
            return result
        generator.enable_fp8_gemm = audited_enable
    status = 'failed'
    try:
        rope.install()
        pointwise.install()
        def trace(stage, **details):
            if stage == 'condition_start':
                rope.clear()
            return original_trace(stage, **details)
        generator._p7_trace = trace
        generator.generate(args)
        status = 'completed'
    finally:
        report = dict(rank=int(os.environ['RANK']), status=status,
                      gpu=prop.name, world_size=2, media_fps=20,
                      generation_target_fps=20, fp8_gemm=bool(args.fp8_gemm),
                      fp8_audit=fp8_audit,
                      audio_extra_lookahead=0.5, audio_history=4,
                      rotary=rope.report(), pointwise=pointwise.report(),
                      runtime_cache=runtime_cache, cache_smoke=cache_smoke,
                      real_time_accepted=False, full_video_quality_accepted=False,
                      live_microphone_accepted=False)
        (Path.cwd() / f'hopper-rank-{os.environ["RANK"]}.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        pointwise.close()
        rope.close()
        generator._p7_trace = original_trace
        generator.StreamingAudioConditioner = original_conditioner
        generator.enable_fp8_gemm = original_enable_fp8
        for hook in fp8_hooks:
            hook.remove()


if __name__ == '__main__':
    main()
