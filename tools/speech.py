"""Authenticated local-model speech web service on a workstation or allocated node."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import threading
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    if 'login' in socket.gethostname().lower() and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Use deploy/speech.sbatch instead of loading models on a login node')
    from tools.models import entries,verify
    for row in entries('speech'):verify(row)
    os.environ['LIVEACT_TTS_VOICE']=os.environ.get('LIVEACT_TTS_VOICE','zf_003')
    from dialogue.supervisor import DialogueSupervisor
    from dialogue.gateway import DialogueState,create_dialogue_app
    from dialogue.public_gateway import GatewayPolicy
    from uvicorn import Config,Server
    run=ROOT/'runs'/f'speech-{uuid4().hex[:12]}'
    supervisor=DialogueSupervisor(ROOT);state=DialogueState(run,supervisor)
    policy=GatewayPolicy.from_environment()
    app,_=create_dialogue_app(state,gateway_policy=policy)
    listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind((args.host,args.port));listener.listen(128)
    port=listener.getsockname()[1]
    server=Server(Config(app,log_level='warning',access_log=False,ws_max_size=8192,
        ws_max_queue=8,timeout_keep_alive=10,proxy_headers=False))
    stop=threading.Event()
    def save(overrides=None):
        report=dict(supervisor.health(),hostname=socket.gethostname(),port=port,
                    run=str(run),path='/dialogue',access_token_file=str(run/'access-token'),
                    default_voice=os.environ['LIVEACT_TTS_VOICE'],
                    updated_at=datetime.now().astimezone().isoformat(),
                    public_dialogue_connected=False,avatar_connected=False,real_time_accepted=False)
        if overrides:report.update(overrides)
        target=run/'service.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temporary.replace(target)
    def status_writer():
        while not stop.is_set():save();stop.wait(1)
    thread=threading.Thread(target=status_writer,daemon=True);thread.start()
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,lambda *a:setattr(server,'should_exit',True))
    print(json.dumps(dict(run=str(run),hostname=socket.gethostname(),port=port,
                         path='/dialogue',access_token_file=str(run/'access-token'))),flush=True)
    try:server.run(sockets=[listener])
    finally:
        stop.set();thread.join(2)
        save(dict(status='stopped',worker_alive=False))
        try:supervisor.close()
        finally:listener.close()


if __name__=='__main__':main()
