"""Explicit full-video numerical candidate around the accepted Hopper entry."""
import json
import os
from pathlib import Path


def main():
    from scripts import hopper_runtime_cache,hopper_cudnn_generate
    if os.environ.get('LIVEACT_HOPPER_NORM_MODE')!='all':raise ValueError('Explicit norm candidate required')
    original=hopper_runtime_cache.pointwise_cache_smoke;sessions=[]
    def initialized(rank,directory):
        report=original(rank,directory)
        import model_liveact.model_memory_sp as model
        from model_liveact.hopper_norm_fused import NormSession
        session=NormSession(model);session.install();sessions.append(session)
        return report
    hopper_runtime_cache.pointwise_cache_smoke=initialized;status='failed'
    try:
        hopper_cudnn_generate.main();status='completed'
    finally:
        reports=[session.report() for session in sessions]
        for session in sessions:session.close()
        hopper_runtime_cache.pointwise_cache_smoke=original
        (Path.cwd()/f'fused-norm-rank-{os.environ["RANK"]}.json').write_text(
            json.dumps(dict(status=status,rank=int(os.environ['RANK']),sessions=reports,
                            human_quality_accepted=False,real_time_accepted=False),indent=2)+'\n')


if __name__=='__main__':main()
