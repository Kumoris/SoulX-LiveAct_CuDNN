"""Pinned preset voices; runtime selection never fetches from the network."""
import hashlib
import json
import os
from pathlib import Path
import re

MANIFEST = Path(__file__).resolve().parents[1] / 'deployment/dialogue/voice-manifest.json'


def voice_catalog():
    return json.loads(MANIFEST.read_text())


def verify_voice(directory, voice_id=None):
    catalog = voice_catalog()
    if voice_id is None:
        voice_id = os.environ.get('LIVEACT_TTS_VOICE', catalog['default_voice'])
    if not isinstance(voice_id, str) or not re.fullmatch(r'z[fm]_[0-9]{3}', voice_id):
        raise ValueError('Choose a configured preset voice ID such as zf_001')
    entries = {entry['id']: entry for entry in catalog['voices']}
    if voice_id not in entries:
        raise ValueError(f'Unconfigured voice {voice_id}; available IDs: {", ".join(entries)}')
    item = entries[voice_id]
    root = Path(directory).resolve()
    path = (root / item['name']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Voice file must stay within its model directory')
    if not path.is_file() or path.stat().st_size != item['size']:
        raise FileNotFoundError(f'Local voice missing or wrong size: {voice_id}; prepare it before inference')
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != item['lfs_sha256']:
        raise ValueError(f'Official preset voice SHA mismatch: {voice_id}')
    return dict(id=voice_id, path=str(path), sha256=sha256,
                repo=catalog['repo'], revision=catalog['revision'])
