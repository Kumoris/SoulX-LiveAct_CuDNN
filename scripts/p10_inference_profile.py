"""Opt-in block diagnostics; never use profiled blocks as throughput evidence.

Set LIVEACT_P10_PROFILE_ROOT to a per-run directory. Both torchrun ranks save
their own trace. Only one warmed block receives operator/module tracing; other
blocks have phase CUDA events without per-phase synchronization.
"""

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import time

import torch


class InferenceProfiler:
    def __init__(self, model):
        root = os.environ.get("LIVEACT_P10_PROFILE_ROOT")
        self.root = Path(root) if root else None
        self.model = model
        self.rank = int(os.environ.get("RANK", "0"))
        self.selected_block = int(os.environ.get("LIVEACT_P10_PROFILE_BLOCK", "3"))
        self.profile = None
        self.handles = []
        self.events = []
        self.block_index = None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _add_module_ranges(self):
        # Module hooks are present only during the one profiled block.
        for layer, block in enumerate(self.model.blocks):
            for name in ("self_attn", "cross_attn", "audio_cross_attn", "ffn"):
                module = getattr(block, name)
                stack = []

                def before(module, arguments, *, stack=stack,
                           label=f"liveact/layer_{layer:02d}/{name}"):
                    scope = torch.profiler.record_function(label)
                    scope.__enter__()
                    stack.append(scope)

                def after(module, arguments, output, *, stack=stack):
                    stack.pop().__exit__(None, None, None)

                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(after, always_call=True))

    def begin_block(self, index):
        if not self.root:
            return
        self.block_index = index
        self.events = []
        self.block_start = time.perf_counter()
        if index == self.selected_block:
            self._add_module_ranges()
            self.profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
            )
            self.profile.__enter__()

    @contextmanager
    def phase(self, name, **details):
        if not self.root:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter()
        start.record()
        scope = (torch.profiler.record_function(f"liveact/{name}")
                 if self.profile else nullcontext())
        try:
            with scope:
                yield
        finally:
            end.record()
            self.events.append((name, details, start, end, time.perf_counter() - tick))

    def end_block(self):
        if not self.root:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.block_start
        rows = [{"phase": name, **details, "cuda_elapsed_ms": start.elapsed_time(end),
                 "cpu_scope_ms": cpu * 1000}
                for name, details, start, end, cpu in self.events]
        record = {"rank": self.rank, "block": self.block_index,
                  "operator_profiled": self.profile is not None,
                  "block_wall_seconds": elapsed, "phases": rows,
                  "note": "CUDA events include stream waits; CPU scopes are enqueue times. Operator totals may overlap and are not wall-time shares."}
        with (self.root / f"phases-rank-{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.profile:
            profile = self.profile
            try:
                profile.__exit__(None, None, None)
                stem = f"block-{self.block_index}-rank-{self.rank}"
                profile.export_chrome_trace(str(self.root / f"{stem}.trace.json"))
                operators = [{"name": item.key, "count": item.count,
                              "self_cpu_us": item.self_cpu_time_total,
                              "self_device_us": item.self_device_time_total,
                              "device_total_us": item.device_time_total}
                             for item in profile.key_averages()]
                operators.sort(key=lambda item: item["self_device_us"], reverse=True)
                cuda_events = sum(str(event.device_type).endswith("CUDA")
                                  for event in profile.events())
                (self.root / f"{stem}.operators.json").write_text(json.dumps({
                    "rank": self.rank, "block": self.block_index,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(),
                    "cuda_device_event_count": cuda_events,
                    "cuda_operator_trace_available": cuda_events > 0,
                    "operators": operators,
                    "note": record["note"],
                }, indent=2) + "\n")
            finally:
                for handle in self.handles:
                    handle.remove()
                self.handles.clear()
                self.profile = None
