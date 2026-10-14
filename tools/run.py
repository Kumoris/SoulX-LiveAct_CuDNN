"""Portable video/resident launch commands, independent of past Slurm evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
import wave

ROOT=Path(__file__).resolve().parents[1]


def prepare_case(input_path,output,root=ROOT):
    root=Path(root).resolve();input_path=Path(input_path)
    if not input_path.is_absolute():input_path=root/input_path
    cases=json.loads(input_path.read_text())
    if not isinstance(cases,list) or len(cases)!=1:raise ValueError('Provide exactly one image/audio/control case')
    case=dict(cases[0])
    for key in ('reference_image','audio'):
        path=Path(case[key]);path=path if path.is_absolute() else root/path
        if not path.is_file():raise FileNotFoundError(str(path))
        case[key]=str(path.resolve())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    audio=Path(case['audio'])
    try:
        with wave.open(str(audio)) as wav:
            ready=(wav.getnchannels(),wav.getsampwidth(),wav.getframerate())==(1,2,16000)
    except (wave.Error,EOFError):ready=False
    if not ready:
        prepared=output/(audio.stem+'-mono16k.wav')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(audio),'-ac','1','-ar','16000',
                        '-c:a','pcm_s16le',str(prepared)],check=True)
        case['audio']=str(prepared.resolve())
    with wave.open(case['audio']) as wav:seconds=wav.getnframes()/16000
    if not 10<=seconds<=15:raise ValueError('Use a 10–15 second sample; do not pad or truncate it')
    from control_spec import normalize_case
    from video_timing import required_blocks,target_frame_count
    normalize_case(case,block_count=required_blocks(target_frame_count(seconds,20)))
    destination=output/'input.json';destination.write_text(json.dumps([case],ensure_ascii=False,indent=2)+'\n')
    return destination


def command(mode,input_path,root=ROOT):
    root=Path(root).resolve()
    entry='scripts.hopper_custom_generate' if mode=='video' else 'scripts.hopper_resident_generate'
    return [str(root/'.hopper-env/bin/python'),'-m','torch.distributed.run','--standalone',
            '--nnodes=1','--nproc_per_node=2','-m',entry,'--ckpt_dir',str(root/'models/LiveAct'),
            '--wav2vec_dir',str(root/'models/chinese-wav2vec2-base'),'--input_json',str(input_path),
            '--size','416*720','--fps','20','--seed','42','--t5_cpu','--dura_print','--no-cudnn-benchmark',
            '--audio_stream_pcm','--audio_lookahead','0.5','--audio_history','4','--fp8_gemm']


def environment(mode,output,template,root=ROOT):
    root=Path(root).resolve();env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join((str(root),str(root/'third_party/LightX2V'))),
               PATH=os.pathsep.join((str(root/'.hopper-env/bin'),str(root/'.liveact-env/bin'),env['PATH'])),
               PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PLATFORM='cuda',USE_CHANNELS_LAST_3D='1',
               TORCHDYNAMO_DISABLE='1',LIVEACT_HOPPER_NORM_MODE='all',LIVEACT_HOPPER_VAE_MODE='overlap',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN')
    for key in ('LIVEACT_P7_TIMELINE','LIVEACT_P10_PROFILE_ROOT','LIVEACT_P10_REPRO_ROOT',
                'LIVEACT_HOPPER_FP8_AUDIT','LIVEACT_R2_SM80_SITE','LIVEACT_R2_SELF_ATTN_MODE'):
        env.pop(key,None)
    if mode=='resident':env.update(LIVEACT_H6_ROOT=str(output),LIVEACT_H6_TEMPLATE=str(template))
    return env


def doctor():
    snapshot=json.loads((ROOT/'source-snapshot.json').read_text())
    import hashlib
    for name,expected in snapshot['copied_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Exported source changed: '+name)
    print('Source snapshot intact; upstream and runtime revisions are recorded in source-snapshot.json.')
    for path in ('.liveact-env/bin/python','.hopper-env/bin/python','.dialogue-env/bin/python',
                 '.liveact-env/bin/sox','third_party/LightX2V/lightx2v'):
        print(path,'OK' if (ROOT/path).exists() else 'NOT INSTALLED')
    print('This checks installation files, not GPU, human quality or sustained realtime acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('doctor','video','resident','preflight'))
    parser.add_argument('--input',default='examples/input.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.mode=='doctor':doctor();return
    if args.mode=='preflight':
        commandline=[str(ROOT/'.hopper-env/bin/python'),'-m','torch.distributed.run','--standalone',
                     '--nproc_per_node=2','-m','scripts.hopper_preflight','--gpu','--require-fp8',
                     '--report',str(ROOT/'runs/hardware.json')]
        (ROOT/'runs').mkdir(exist_ok=True)
        env=environment('video',ROOT/'runs',None)
        return subprocess.run(commandline,cwd=ROOT,env=env,check=True)
    if not args.dry_run:
        from tools.models import entries,verify
        for row in entries('avatar'):verify(row)
        if not (ROOT/'third_party/LightX2V/lightx2v').is_dir():raise FileNotFoundError('Run bootstrap --mode avatar first')
    output=args.output or ROOT/'runs'/f'{args.mode}-{uuid4().hex[:12]}'
    if not output.is_absolute():output=ROOT/output
    prepared=prepare_case(args.input,output)
    invocation=command(args.mode,prepared);env=environment(args.mode,output,prepared)
    (output/'launch.json').write_text(json.dumps(dict(command=invocation,mode=args.mode,
        root=str(ROOT),model_math_changed=False,public_dialogue_connected=False,real_time_accepted=False),indent=2)+'\n')
    print(json.dumps(dict(mode=args.mode,output=str(output),command=invocation),ensure_ascii=False),flush=True)
    if args.dry_run:return
    process=subprocess.Popen(invocation,cwd=output,env=env)
    try:
        result=process.wait()
        if result:raise subprocess.CalledProcessError(result,invocation)
    except KeyboardInterrupt:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise


if __name__=='__main__':main()
