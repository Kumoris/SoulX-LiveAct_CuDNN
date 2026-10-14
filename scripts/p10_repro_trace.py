"""Opt-in tensor/RNG fingerprints; synchronizes CUDA, never measures throughput."""

import hashlib
import json
import os
from pathlib import Path


class ReproTrace:
    def __init__(self, path=None, *, device=None, settings=None):
        self.path = Path(path) if path else None
        self.device = device
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x') as stream:
                stream.write(json.dumps({'stage': 'settings', 'settings': settings or {},
                    'diagnostic_only': True,
                    'limitation': 'Tensor reads synchronize CUDA and perturb execution; not performance evidence.'}) + '\n')

    @classmethod
    def from_environment(cls, *, case_index, rank, device, settings):
        root = os.environ.get('LIVEACT_P10_REPRO_ROOT')
        path = Path(root) / f'case-{case_index}-rank-{rank}.jsonl' if root else None
        return cls(path, device=device, settings=settings)

    def _write(self, row):
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')

    @staticmethod
    def fingerprint(value):
        import torch
        tensor = torch.as_tensor(value).detach().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(raw).hexdigest()}

    def tensor(self, stage, value, *, block=None, step=None):
        if self.path is None:
            return
        self._write({'stage': stage, 'block': block, 'step': step,
                     'tensor': self.fingerprint(value)})

    def rng(self, stage, *, block=None, step=None):
        if self.path is None:
            return
        import torch
        states = {'cpu': self.fingerprint(torch.get_rng_state())}
        if self.device is not None:
            target = torch.device('cuda', self.device) if isinstance(self.device, int) else torch.device(self.device)
            if target.type == 'cuda':
                states['cuda'] = self.fingerprint(torch.cuda.get_rng_state(target))
        self._write({'stage': stage, 'block': block, 'step': step, 'rng': states})


def compare_traces(left, right):
    """Find the first differing common tensor/RNG stage in execution order."""
    def read(path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        events = [row for row in rows if 'tensor' in row or 'rng' in row]
        keys = [(row['stage'], row.get('block'), row.get('step')) for row in events]
        if len(set(keys)) != len(keys) or not events:
            raise ValueError('Duplicate or missing fingerprint events')
        return events, keys
    a, ak = read(left)
    b, bk = read(right)
    if ak != bk:
        raise ValueError('Trace stages/order differ; cannot compare')
    differences = []
    for x, y in zip(a, b):
        field = 'tensor' if 'tensor' in x else 'rng'
        if x.get(field) != y.get(field):
            differences.append({'stage': x['stage'], 'block': x.get('block'),
                'step': x.get('step'), 'kind': field, 'left': x.get(field), 'right': y.get(field)})
    return {'compared_events': len(a), 'exact_matching_events': len(a) - len(differences),
            'all_events_exact': not differences,
            'first_difference': differences[0] if differences else None,
            'differing_stages': [{'stage': row['stage'], 'block': row['block'], 'step': row['step']}
                                 for row in differences]}
