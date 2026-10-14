"""Opt-in SP2 Hopper output scheduling; retain the accepted generator math."""
import json
import os
from pathlib import Path


def main():
    # Defer every Torch import until hopper_generate configures its fresh cache.
    from scripts import hopper_runtime_cache, hopper_generate
    mode = os.environ.get('LIVEACT_HOPPER_VAE_MODE')
    if mode not in ('serial', 'overlap'):
        raise ValueError('Select serial or overlap explicitly')
    if any(os.environ.get(key) for key in ('LIVEACT_P7_TIMELINE', 'LIVEACT_P10_PROFILE_ROOT',
                                          'LIVEACT_P10_REPRO_ROOT', 'LIVEACT_HOPPER_FP8_AUDIT')):
        raise ValueError('Output scheduling comparison requires diagnostic hooks off')
    original_smoke = hopper_runtime_cache.pointwise_cache_smoke
    adapters, pipelines = [], []

    def initialized_smoke(rank, directory):
        report = original_smoke(rank, directory)
        # The unmodified Hopper entry has now verified the GPU and the cache.
        import generate as generator
        from model_liveact.bounded_video_pipeline import BoundedVideoPipeline
        from model_liveact.vae_overlap import CUDAOverlapBackend, GeneratorOverlapAdapter

        def create(vae, gpu_rank, target_frames, blocks):
            if target_frames != 300 or blocks != 10:
                raise ValueError('Use the fixed, accepted 15-second fixture')
            backend = CUDAOverlapBackend(vae, rank=gpu_rank, world_size=2)
            pipeline = BoundedVideoPipeline(backend, mode=mode,
                                           target_frames=target_frames, expected_blocks=blocks)
            pipelines.append(pipeline)
            return pipeline

        adapter = GeneratorOverlapAdapter(generator, create)
        adapter.install()
        adapters.append(adapter)
        return report

    hopper_runtime_cache.pointwise_cache_smoke = initialized_smoke
    status = 'failed'
    try:
        hopper_generate.main()
        status = 'completed'
    finally:
        report = dict(status=status, rank=int(os.environ.get('RANK', '0')), mode=mode,
                      pipelines=[dict(**pipeline.report(), backend=pipeline.backend.report())
                                 for pipeline in pipelines],
                      only_factor_changed='VAE local-decode and D2H scheduling',
                      weights_steps_wave_and_audio_changed=False, real_time_accepted=False)
        for pipeline in pipelines:
            pipeline.close()
        for adapter in adapters:
            adapter.close()
        hopper_runtime_cache.pointwise_cache_smoke = original_smoke
        (Path.cwd()/f'hopper-output-rank-{report["rank"]}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
