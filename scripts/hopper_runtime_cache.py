"""Keep cold Triton compilation off shared home directories and other ranks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile


def _filesystem_type(path: Path) -> str | None:
    """Identify the actual Linux mount behind /tmp, including symlinks."""
    try:
        mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    except OSError:
        return None
    resolved = path.resolve()
    candidates = []
    for line in mounts:
        location, filesystem = line.split(' - ', 1)
        mount = Path(location.split()[4].replace(r'\040', ' '))
        if resolved.is_relative_to(mount):
            candidates.append((len(str(mount)), filesystem.split()[0]))
    return max(candidates)[1] if candidates else None


def configure_rank_cache(report_dir: Path) -> dict:
    """Call in each torchrun child before importing torch or Triton."""
    if 'torch' in sys.modules or 'triton' in sys.modules:
        raise RuntimeError('Configure Hopper runtime caches before importing torch/Triton')
    rank = int(os.environ['RANK'])
    job = os.environ.get('SLURM_JOB_ID', 'standalone')
    # Ignore a possibly shared TMPDIR inherited from the login environment.
    root = Path(tempfile.mkdtemp(prefix=f'liveact-hopper-{job}-rank{rank}-', dir='/tmp'))
    filesystem = _filesystem_type(root)
    if filesystem in {'nfs', 'nfs4', 'cifs', 'lustre', 'gpfs', 'beegfs', 'ceph', 'fuse.sshfs'}:
        raise RuntimeError(f'Hopper runtime cache requires node-local /tmp, found {filesystem}')
    directories = {}
    for variable, folder in (
        ('TRITON_CACHE_DIR', 'triton'),
        ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
        ('CUDA_CACHE_PATH', 'cuda'),
    ):
        path = root / folder
        path.mkdir(mode=0o700)
        os.environ[variable] = str(path)
        directories[variable] = str(path)
    report = dict(status='fresh_rank_cache_configured', rank=rank, job_id=job,
                  hostname=socket.gethostname(), pid=os.getpid(),
                  root=str(root), directories=directories,
                  configured_before_torch_and_triton=True,
                  filesystem_type=filesystem,
                  local_tmp_filesystem_verification_pending=filesystem is None)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f'hopper-cache-rank-{rank}.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print('Hopper runtime cache: ' + json.dumps(report), flush=True)
    return report


def pointwise_cache_smoke(local_rank: int, report_dir: Path) -> dict:
    """Compile both fused operators on each GPU before loading model weights."""
    import torch
    import triton
    from model_liveact.pointwise_fused import modulation, residual

    device = torch.device('cuda', local_rank)
    # An independent RNG preserves the animation's seeded input stream.
    rng = torch.Generator(device=device).manual_seed(42)
    cases = []
    with torch.inference_mode():
        for tokens in (3510, 4680):
            for parameter_dtype in (torch.bfloat16, torch.float32):
                x = torch.randn(1, tokens, 5120, device=device,
                                dtype=torch.bfloat16, generator=rng)
                y = torch.randn(x.shape, device=device, dtype=x.dtype, generator=rng)
                shift = torch.randn(1, 1, 5120, device=device,
                                    dtype=parameter_dtype, generator=rng)
                scale = torch.randn(shift.shape, device=device,
                                    dtype=parameter_dtype, generator=rng)
                for name, call, reference in (
                    ('modulation', lambda: modulation(x, shift, scale),
                     (x.float() * (1 + scale) + shift).to(x.dtype)),
                    ('residual', lambda: residual(x, y, scale),
                     (x + y * scale).to(x.dtype)),
                ):
                    # First call compiles; second checks the cached kernel.
                    for invocation in range(2):
                        actual = call()
                        if not torch.equal(actual, reference):
                            raise ValueError(f'Hopper {name} BF16 mismatch: '
                                             f'{tokens}, {parameter_dtype}, call {invocation}')
                    cases.append(dict(operator=name, tokens=tokens,
                                      parameter_dtype=str(parameter_dtype),
                                      repeated_outputs_exact=True))
                del x, y, shift, scale, reference, actual
    torch.cuda.synchronize(local_rank)
    report = dict(status='pointwise_cold_and_cached_kernels_passed',
                  rank=int(os.environ['RANK']), triton_version=triton.__version__,
                  cases=cases, weights_loaded=False, real_time_accepted=False)
    (report_dir / f'hopper-cache-smoke-rank-{os.environ["RANK"]}.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print('Hopper cache smoke: ' + json.dumps(report), flush=True)
    return report
