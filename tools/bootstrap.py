"""Build separate environments with pinned runtime versions, at any project path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(command,**kwargs):
    print(' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('avatar','speech','all'),default='all')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,10):raise ValueError('Use Python 3.10 to match the validated runtime')
    if 'login' in os.uname().nodename.lower() and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Compile/install on an allocated node; use deploy/setup.sbatch')
    base=ROOT/'.liveact-env'
    if not base.exists():run([sys.executable,'-m','venv',base])
    python=base/'bin/python'
    index='https://download.pytorch.org/whl/'+('cpu' if args.mode=='speech' else 'cu128')
    run([python,'-m','pip','install','--index-url',index,'torch==2.8.0','torchaudio==2.8.0','torchvision==0.23.0'])
    run([python,'-m','pip','install','-r',ROOT/'requirements/base.txt'])
    for binary in ('sox','ffmpeg','ffprobe'):
        executable=shutil.which(binary)
        if not executable:raise FileNotFoundError('Install '+binary+' with conda or your system package manager')
        target=base/'bin'/binary
        if not target.exists():target.symlink_to(Path(executable).resolve())
    if args.mode in ('avatar','all'):
        run([python,'-m','pip','install','-r',ROOT/'requirements/avatar.txt','-c',ROOT/'requirements/constraints.txt'])
        light=ROOT/'third_party/LightX2V';light.parent.mkdir(exist_ok=True)
        revision=json.loads((ROOT/'source-snapshot.json').read_text())['lightx2v_commit']
        if not light.exists():run(['git','clone','https://github.com/ModelTC/LightX2V.git',light])
        actual=subprocess.check_output(['git','-C',str(light),'rev-parse','HEAD'],text=True).strip()
        if actual!=revision:run(['git','-C',light,'checkout',revision])
        overlay=ROOT/'.hopper-env'
        if not overlay.exists():run([python,'-m','venv','--system-site-packages',overlay])
        run([overlay/'bin/python','-m','pip','install','-r',ROOT/'requirements/hopper.txt',
             '-c',ROOT/'requirements/constraints.txt'])
        run([overlay/'bin/python','-m','pip','check'])
    if args.mode in ('speech','all'):
        overlay=ROOT/'.dialogue-env'
        if not overlay.exists():run([python,'-m','venv','--system-site-packages',overlay])
        run([overlay/'bin/python','-m','pip','install','-r',ROOT/'requirements/speech.txt',
             '-c',ROOT/'requirements/constraints.txt'])
        env=os.environ.copy();env.update(CMAKE_ARGS='-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF',
                                         CMAKE_BUILD_PARALLEL_LEVEL='4')
        run([overlay/'bin/python','-m','pip','install','--no-build-isolation','llama-cpp-python==0.3.16'],env=env)
        run([overlay/'bin/python','-m','pip','check'])
    print('Environments ready. Transfer/download weights explicitly, then run tools.models verify.')


if __name__=='__main__':main()
