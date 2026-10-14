"""Explicit weight download and streaming SHA verification; inference stays offline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]


def entries(group):
    rows=[]
    if group in ('avatar','all'):
        for item in json.loads((ROOT/'deployment/model-manifest.json').read_text())['files']:
            directory='models/LiveAct' if item['repo']=='Soul-AILab/LiveAct' else 'models/chinese-wav2vec2-base'
            rows.append(dict(item,name=str(Path(item['path']).relative_to(directory)),directory=directory,
                             source='modelscope',revision=item['revision_requested']))
    if group in ('speech','all'):
        models=json.loads((ROOT/'deployment/dialogue/model-manifest.json').read_text())['models']
        for model in models:
            for item in model['files']:
                rows.append(dict(item,path=str(Path(model['directory'])/item['name']),
                                 directory=model['directory'],repo=model['repo'],revision=model['revision'],
                                 source='huggingface',sha256=item.get('sha256') or item.get('lfs_sha256')))
        voices=json.loads((ROOT/'deployment/dialogue/voice-manifest.json').read_text())
        present={row['path'] for row in rows}
        for voice in voices['voices']:
            path=str(Path(voices['directory'])/voice['name'])
            if path not in present:
                rows.append(dict(voice,path=path,directory=voices['directory'],repo=voices['repo'],
                                 revision=voices['revision'],source='huggingface',sha256=voice['lfs_sha256']))
    return rows


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def verify(row,root=ROOT):
    relative=Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Model path escaped project')
    # Model directories can intentionally point to a verified existing copy.
    # Integrity is still checked against the frozen file size and SHA.
    path=(Path(root)/relative).resolve()
    if not path.is_file() or path.stat().st_size!=row['size']:
        raise ValueError('Model missing or wrong size: '+row['path'])
    actual=digest(path)
    if row.get('sha256') and actual!=row['sha256']:
        raise ValueError('Model SHA256 mismatch: '+row['path'])
    return dict(path=row['path'],size=row['size'],sha256=actual,expected_sha256_checked=bool(row.get('sha256')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('download','verify'))
    parser.add_argument('--group',choices=('avatar','speech','all'),default='all')
    args=parser.parse_args();verified=[]
    for row in entries(args.group):
        path=ROOT/row['path']
        if args.action=='download':
            try:
                verified.append(verify(row));print('already verified',row['path'],flush=True);continue
            except ValueError:pass
            path.parent.mkdir(parents=True,exist_ok=True)
            if row['source']=='huggingface':
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=row['repo'],filename=row['name'],revision=row['revision'],
                                local_dir=str(ROOT/row['directory']))
            else:
                url=f"https://modelscope.cn/models/{row['repo']}/resolve/{row['revision']}/{quote(row['name'])}"
                subprocess.run(['aria2c','--continue=true','--file-allocation=none',
                    '--max-connection-per-server=8','--split=8','--max-tries=10','--retry-wait=5',
                    '--auto-file-renaming=false','--allow-overwrite=true',
                    '--checksum=sha-256='+row['sha256'],'--dir='+str(path.parent),'--out='+path.name,url],check=True)
        verified.append(verify(row));print('verified',row['path'],flush=True)
    report=dict(status='files_verified',group=args.group,files=verified,real_time_accepted=False)
    output=ROOT/'runs/models';output.mkdir(parents=True,exist_ok=True)
    (output/f'{args.group}-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],files=len(verified),group=args.group)))


if __name__=='__main__':main()
