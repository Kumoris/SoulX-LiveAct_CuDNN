"""Custom 10–15 s inputs using the frozen Hopper math and two-slot decoder.

The benchmark entry still requires exactly 300 frames. This separate entry
allows the existing decoder to flush the actual custom EOS without padding
the user's audio or changing the benchmark's source checksums.
"""
import json
import os
from pathlib import Path

from video_timing import required_blocks


def validate_layout(frames, blocks):
    if (type(frames) is not int or not 200 <= frames <= 300
            or type(blocks) is not int or blocks != required_blocks(frames)):
        raise ValueError('Custom media must retain 20 FPS and a 10–15 second EOS')


def custom_overlap_main():
    # No Torch import before hopper_generate creates each rank's local cache.
    from scripts import hopper_runtime_cache, hopper_generate

    if os.environ.get('LIVEACT_HOPPER_VAE_MODE') != 'overlap':
        raise ValueError('Custom media requires the original two-slot overlap')
    if any(os.environ.get(key) for key in (
            'LIVEACT_P7_TIMELINE', 'LIVEACT_P10_PROFILE_ROOT',
            'LIVEACT_P10_REPRO_ROOT', 'LIVEACT_HOPPER_FP8_AUDIT')):
        raise ValueError('Keep diagnostic hooks off for custom media')
    original = hopper_runtime_cache.pointwise_cache_smoke
    adapters, pipelines = [], []

    def initialized(rank, directory):
        report = original(rank, directory)
        import generate as generator
        from model_liveact.bounded_video_pipeline import BoundedVideoPipeline
        from model_liveact.vae_overlap import CUDAOverlapBackend, GeneratorOverlapAdapter

        def create(vae, gpu_rank, target_frames, blocks):
            validate_layout(target_frames, blocks)
            backend = CUDAOverlapBackend(vae, rank=gpu_rank, world_size=2)
            pipeline = BoundedVideoPipeline(backend, mode='overlap',
                target_frames=target_frames, expected_blocks=blocks)
            pipelines.append(pipeline)
            return pipeline

        adapter = GeneratorOverlapAdapter(generator, create)
        adapter.install()
        adapters.append(adapter)
        return report

    hopper_runtime_cache.pointwise_cache_smoke = initialized
    status = 'failed'
    try:
        hopper_generate.main()
        status = 'completed'
    finally:
        report = dict(status=status, rank=int(os.environ['RANK']), mode='overlap',
            pipelines=[dict(**pipeline.report(), backend=pipeline.backend.report())
                       for pipeline in pipelines],
            decoder_math_changed=False, custom_media=True,
            human_quality_accepted=False, real_time_accepted=False)
        try:
            for pipeline in pipelines:
                pipeline.close()
        finally:
            for adapter in adapters:
                adapter.close()
            hopper_runtime_cache.pointwise_cache_smoke = original
            (Path.cwd()/f'hopper-output-rank-{report["rank"]}.json').write_text(
                json.dumps(report, indent=2)+'\n')


def main():
    from scripts import hopper_vae_overlap, hopper_cudnn_generate, hopper_norm_generate

    mode = os.environ.get('LIVEACT_HOPPER_NORM_MODE')
    if mode not in ('off', 'all'):
        raise ValueError('Explicit normalization mode required')
    original = hopper_vae_overlap.main
    hopper_vae_overlap.main = custom_overlap_main
    try:
        if mode == 'all':
            hopper_norm_generate.main()
        else:
            hopper_cudnn_generate.main()
    finally:
        hopper_vae_overlap.main = original


if __name__ == '__main__':
    main()
