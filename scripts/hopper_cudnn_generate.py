"""Native attention candidate around unmodified Hopper/ordered VAE entries."""
import json
import os
from pathlib import Path


def main():
    from scripts import hopper_runtime_cache,hopper_vae_overlap
    original=hopper_runtime_cache.pointwise_cache_smoke
    sessions=[]
    def initialized(rank,directory):
        report=original(rank,directory)
        import model_liveact.model_memory_sp as model
        from model_liveact.native_cudnn_attention import NativeCudnnSession
        session=NativeCudnnSession(model);session.install();sessions.append(session)
        return report
    hopper_runtime_cache.pointwise_cache_smoke=initialized
    try:
        hopper_vae_overlap.main()
    finally:
        reports=[session.report() for session in sessions]
        for session in sessions:session.close()
        hopper_runtime_cache.pointwise_cache_smoke=original
        (Path.cwd()/f'cudnn-attention-rank-{os.environ["RANK"]}.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
