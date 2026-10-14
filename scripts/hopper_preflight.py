"""Validate portable fixtures locally, or actual dual-Hopper hardware via torchrun.

This gate never loads the animation model or accepts real-time performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_device(name: str, capability: tuple[int, int], memory: int) -> None:
    if not any(model in name.upper() for model in ("H100", "H200")):
        raise ValueError("H100/H200 required; the A800 route is abandoned")
    if tuple(capability) != (9, 0):
        raise ValueError("Actual Hopper SM90 device required")
    if memory < 75 * 2**30:
        raise ValueError("This deployment targets at least 75 GiB per Hopper GPU")


def validate_pair(devices: list[dict]) -> None:
    if len(devices) != 2 or {d["rank"] for d in devices} != {0, 1}:
        raise ValueError("Two actual GPU ranks required")
    for device in devices:
        validate_device(device["name"], device["capability"], device["memory_bytes"])
    if len({d["hostname"] for d in devices}) != 1:
        raise ValueError("Same-node GPU pair required")
    if len({d["name"] for d in devices}) != 1:
        raise ValueError("Use a matching H100 pair or a matching H200 pair")


def offline() -> dict:
    manifest = json.loads((ROOT / "deployment/hopper/references/manifest.json").read_text())
    for item in manifest["references"] + manifest["inputs"]:
        path = ROOT / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Migration fixture changed: {item['path']}")
    models = json.loads((ROOT / "deployment/model-manifest.json").read_text())
    for item in models["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Model file missing or incorrect size: {item['path']}")
    return dict(status="portable_references_and_model_sizes_passed",
                reference_videos=len(manifest["references"]), model_files=len(models["files"]),
                full_model_sha256_on_destination_required=True, hardware_validated=False)


def gpu(*, require_fp8: bool) -> dict:
    import socket
    import torch
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise ValueError("Run with torchrun --nproc_per_node=2 on the Hopper node")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    prop = torch.cuda.get_device_properties(local_rank)
    device = dict(rank=rank, hostname=socket.gethostname(), name=prop.name,
                  capability=list(torch.cuda.get_device_capability(local_rank)),
                  memory_bytes=prop.total_memory)
    validate_device(prop.name, tuple(device["capability"]), prop.total_memory)
    dist.init_process_group("nccl")
    try:
        devices = [None, None]
        dist.all_gather_object(devices, device)
        validate_pair(devices)
        value = torch.tensor([rank + 1.0], device="cuda", dtype=torch.float32)
        dist.all_reduce(value)
        if value.item() != 3.0:
            raise ValueError("NCCL reduction failed")
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        layer = torch.nn.Linear(256, 128).to(device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            reference = layer(x)
            if not torch.isfinite(reference).all():
                raise ValueError("BF16 GEMM produced nonfinite values")
            fp8_result = None
            if require_fp8:
                from fp8_gemm import FP8GemmOptions, FP8Linear
                fp8 = FP8Linear(layer, options=FP8GemmOptions(fp16_weight_storage="keep"))
                output = fp8(x)
                if fp8._fp8_weight is None or not torch.isfinite(output).all():
                    raise ValueError("FP8 weight materialization or kernel output failed")
                relative_l2 = ((output.float() - reference.float()).norm()
                               / reference.float().norm().clamp_min(1e-8)).item()
                if relative_l2 > 0.1:
                    raise ValueError("FP8 kernel smoke error exceeded 0.1 relative L2")
                fp8_result = dict(relative_l2=relative_l2,
                                  weight_dtype=str(fp8._fp8_weight.dtype),
                                  operator=type(fp8._fp8_linear_op).__name__,
                                  scope="Tiny kernel smoke only; model quality and throughput untested")
        dist.barrier()
        return dict(status="dual_hopper_nccl_and_kernel_smoke_passed", devices=devices,
                    torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                    nccl_version=torch.cuda.nccl.version(), fp8_kernel=fp8_result,
                    hardware_validated=True, weights_loaded=False)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--require-fp8", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.require_fp8 and not args.gpu:
        parser.error("--require-fp8 requires --gpu")
    report = offline() if args.offline else gpu(require_fp8=args.require_fp8)
    report.update(checked_at=datetime.now().astimezone().isoformat(),
                  generation_target_fps=20, media_fps=20,
                  real_time_accepted=False, full_model_quality_accepted=False)
    if int(os.environ.get("RANK", "0")) == 0:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
