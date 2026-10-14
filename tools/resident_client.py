"""Private renderer client: bounded PCM input, explicit EOS and terminal cleanup."""
import argparse
import base64
import json
from pathlib import Path
import time
from uuid import uuid4
import wave

from scripts.hopper_resident_transport import request


def wait_turn(root,identifier,timeout=300):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=request(root,'status',turn_id=identifier)
        render=state.get('render')
        if state['status'] in ('completed','cancelled') and isinstance(render,dict):return state
        time.sleep(.1)
    raise TimeoutError('Renderer did not finish cleanup before the transition deadline')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('status','send','cancel','shutdown'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audio',type=Path)
    parser.add_argument('--turn-id')
    parser.add_argument('--paced',action='store_true')
    args=parser.parse_args()
    if args.action=='send':
        if args.audio is None:parser.error('send requires --audio')
        identifier=args.turn_id or 'turn-'+uuid4().hex[:12]
        with wave.open(str(args.audio)) as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,16000):
                raise ValueError('Provide mono PCM16/16kHz WAV')
            if not 10<=wav.getnframes()/16000<=15:raise ValueError('Use a 10–15 second sample')
            request(args.root,'start',turn_id=identifier)
            start=time.monotonic();samples=0;sequence=0
            try:
                while packet:=wav.readframes(1600):
                    if args.paced:
                        delay=start+samples/16000-time.monotonic()
                        if delay>0:time.sleep(delay)
                    request(args.root,'pcm',turn_id=identifier,sequence=sequence,
                            pcm=base64.b64encode(packet).decode())
                    samples+=len(packet)//2;sequence+=1
                request(args.root,'eos',turn_id=identifier)
                result=wait_turn(args.root,identifier)
            except BaseException:
                request(args.root,'cancel',turn_id=identifier)
                raise
    elif args.action=='cancel':
        if not args.turn_id:parser.error('cancel requires --turn-id')
        result=request(args.root,'cancel',turn_id=args.turn_id)
    else:result=request(args.root,args.action)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
