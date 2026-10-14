"""Ordered two-slot output scheduling, with completed-output wall timestamps."""
from collections import deque
import threading
import time


class BoundedVideoPipeline:
    def __init__(self, backend, *, mode, target_frames, expected_blocks):
        if mode not in ('serial', 'overlap') or target_frames < 1 or expected_blocks < 1:
            raise ValueError('Invalid bounded output configuration')
        self.backend = backend
        self.mode = mode
        self.target_frames = target_frames
        self.expected_blocks = expected_blocks
        self.owner = threading.get_ident()
        self.pending = deque()
        self.starts = {}
        self.completed = []
        self.submitted = 0
        self.max_pending = 0
        self.generated_frames = 0
        self.effective_frames = 0
        self.closed = False

    def _main(self):
        if threading.get_ident() != self.owner:
            raise ValueError('Output scheduling and collectives require their owner thread')
        if self.closed:
            raise ValueError('Output pipeline closed')

    def begin_block(self, block):
        self._main()
        if block != len(self.starts) or block >= self.expected_blocks:
            raise ValueError('Compute blocks must begin in order')
        self.starts[block] = time.perf_counter()

    def consume(self, block, latent, outputs):
        self._main()
        if block != self.submitted or block not in self.starts or len(self.pending) >= 2:
            raise ValueError('Output order or two-slot bound violated')
        submitted = time.perf_counter()
        job = self.backend.submit(block, latent, asynchronous=self.mode == 'overlap')
        self.pending.append((block, job, submitted))
        self.submitted += 1
        self.max_pending = max(self.max_pending, len(self.pending))
        # The first block is never held back to fill a pipeline.
        if self.mode == 'serial' or block == 0 or len(self.pending) == 2:
            self._complete(outputs)

    def _complete(self, outputs):
        block, job, submitted = self.pending[0]
        if block != len(self.completed):
            raise ValueError('Decoded outputs out of order')
        value = self.backend.finish(job)
        count = int(value.shape[2])
        if count != (21 if block == 0 else 32):
            raise ValueError('First/later decoded frame count changed')
        if getattr(value, 'is_cuda', False):
            raise ValueError('Output completion must include D2H')
        done = time.perf_counter()
        effective = min(count, max(0, self.target_frames - self.effective_frames))
        self.completed.append(dict(block=block, compute_start=self.starts[block],
            submitted_at=submitted, completed_at=done, decoded_frames=count,
            effective_frames=effective, pts_start_seconds=self.effective_frames/20,
            pts_end_seconds=(self.effective_frames+effective)/20,
            compute_start_to_output_seconds=done-self.starts[block],
            pending_before_completion=len(self.pending)))
        self.generated_frames += count
        self.effective_frames += effective
        outputs.append(value)
        self.pending.popleft()

    def flush(self, outputs):
        self._main()
        while self.pending:
            self._complete(outputs)
        if self.submitted != self.expected_blocks or self.effective_frames != self.target_frames:
            raise ValueError('EOS flush missing blocks or frames')

    def report(self):
        return dict(mode=self.mode, submitted_blocks=self.submitted,
                    completed_blocks=len(self.completed), max_pending=self.max_pending,
                    pending=len(self.pending), generated_frames=self.generated_frames,
                    effective_frames=self.effective_frames, outputs=self.completed,
                    first_block_immediate=True, max_pending_limit=2)

    def close(self):
        if not self.closed:
            self.backend.close()
            self.pending.clear()
            self.closed = True
