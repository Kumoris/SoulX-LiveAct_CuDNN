"""Resident SP2 Hopper worker with open PCM, owner collectives and task resets."""
import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class ChunkWriter:
    def __init__(self, turn):
        self.turn=turn;self.queue=queue.Queue(maxsize=2);self.error=None
        self.peak_pending=0
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:return
                if self.turn.queue.cancelled:continue
                row,pixels=item
                destination=self.turn.directory/f"chunk-{row['block']:03d}.mp4"
                frames,height,width,_=pixels.shape
                command=['ffmpeg','-v','error','-nostdin','-y',
                    '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','pipe:0',
                    '-f','s16le','-ar','16000','-ac','1','-ss',str(row['pts_start_seconds']),
                    '-t',str(frames/20),'-i',str(self.turn.directory/'audio.pcm'),
                    '-map','0:v:0','-map','1:a:0','-c:v','libx264','-threads','2',
                    '-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-bf','0',
                    '-c:a','aac','-movflags','+faststart',str(destination)]
                subprocess.run(command,input=pixels.tobytes(),stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,check=True,timeout=30)
                if self.turn.queue.cancelled:continue
                result=dict(block=row['block'],frames=frames,start_pts=row['pts_start_seconds'],
                    end_pts=row['pts_end_seconds'],cpu_ready_at=row['completed_at'],
                    readable_at=time.monotonic(),path=str(destination),turn_id=self.turn.id,
                    input_snapshot=row['input_snapshot'])
                self.turn.chunks.append(result)
                with (self.turn.directory/'chunks.jsonl').open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
        except BaseException as error:self.error=error

    def offer(self,tensor,row):
        if self.error is not None:raise RuntimeError('Chunk writer failed') from self.error
        if self.turn.queue.cancelled:return
        import numpy as np
        # Same BF16 intermediate conversion as the accepted full-file export.
        pixels=((tensor.permute(0,2,3,4,1)[0,:row['effective_frames']]+1.0)/2).float().numpy()
        pixels=(pixels*255).astype(np.uint8)
        self.queue.put((row,pixels),timeout=15)
        self.peak_pending=max(self.peak_pending,self.queue.qsize())

    def close(self):
        if self.thread.is_alive():self.queue.put(None,timeout=35)
        self.thread.join(timeout=40)
        if self.thread.is_alive():raise TimeoutError('Chunk writer did not stop')
        if self.error is not None:raise RuntimeError('Chunk writer failed') from self.error


class Runtime:
    def __init__(self, generator, rank, root, template):
        from scripts.hopper_resident_transport import Coordinator
        self.generator,self.rank,self.root=generator,rank,Path(root).resolve()
        self.coordinator=Coordinator(self.root,template) if rank==0 else None
        self.group=None;self.turn=None;self.source=None;self.pipeline=None;self.writer=None
        self.cancelled=False;self.cancel_reason=None
        self.model_loads=0;self.model_ready_count=0
        self.tasks=[];self.cache={};self.cache_hits={'text':0,'image':0,'vae':0}
        self.cache_misses={'text':0,'image':0,'vae':0}
        self.last_audio=None;self.audio_snapshots={}

    def broadcast(self,payload):
        values=[payload if self.rank==0 else None]
        self.generator.dist.broadcast_object_list(values,src=0,group=self.group)
        return values[0]

    def trace(self,stage,**details):
        if stage=='model_load_start':self.model_loads+=1
        if stage=='model_ready':self.model_ready_count+=1

    def cases(self, unused_input_path):
        self.group=self.generator.dist.new_group(backend='gloo')
        if self.model_loads!=1 or self.model_ready_count!=1:
            raise ValueError('Weights must load once before serving resident turns')
        if self.rank==0:
            self.coordinator.model_loads=self.model_loads
            self.coordinator.model_ready=True;self.coordinator.save_service()
        while True:
            if self.rank==0:
                turn=self.coordinator.next_turn()
                header=(dict(data=turn.data,id=turn.id,directory=str(turn.directory)) if turn else None)
            else:header=None
            header=self.broadcast(header)
            if header is None:return
            self.turn=header
            data=copy.deepcopy(header['data']);data['_h6_turn_id']=header['id']
            yield data

    def start(self,index,data,cache):
        from scripts.p10_resident_adapter import reset_kv
        self.cancelled=False;self.cancel_reason=None;self.pipeline=None;self.last_audio=None
        self.audio_snapshots={}
        self.turn_start=time.monotonic();self.index=index
        cleared=reset_kv(cache)
        if cleared!=240:raise ValueError('Reset all 40 layers and three timesteps on both ranks')
        self.generator.torch.cuda.synchronize()
        self.cleared=cleared
        self.hits_before=dict(self.cache_hits);self.misses_before=dict(self.cache_misses)
        os.chdir(self.turn['directory'])
        self.writer=ChunkWriter(self.coordinator.current) if self.rank==0 else None

    def cached(self,name,builder):
        if name not in self.cache:
            self.cache[name]=builder();self.cache_misses[name]+=1
        else:self.cache_hits[name]+=1
        return self.cache[name]

    @staticmethod
    def build_image(Image,transform,path,clip,device,torch):
        image=Image.open(path).convert('RGB')
        cond_image=transform(image).unsqueeze(1).unsqueeze(0).to(device,torch.bfloat16)
        clip.model.to(device)
        clip_context=clip.visual(cond_image)
        clip.model.cpu()
        # Same cleanup used by the accepted initial image path.
        from generate import torch_gc
        torch_gc()
        return cond_image,clip_context

    def audio(self,path,fps):
        from src.audio_analysis.open_pcm import OpenPreparedPCM,TurnCancelled
        runtime=self
        rank_source=(OpenPreparedPCM(self.coordinator.current.queue,fps=fps,
            sox=Path(self.generator.__file__).resolve().parent/'.liveact-env/bin/sox') if self.rank==0 else None)
        class DistributedSource:
            source_eof=False
            source_samples=0
            def window(self,*args,**kwargs):
                payload=None
                if runtime.rank==0:
                    try:payload=dict(kind='window',value=rank_source.window(*args,**kwargs))
                    except TurnCancelled as error:payload=dict(kind='cancel',reason=str(error))
                    except BaseException as error:payload=dict(kind='fatal',reason=f'{type(error).__name__}: {error}')
                payload=runtime.broadcast(payload)
                if payload['kind']=='cancel':
                    runtime.cancel_reason=payload['reason']
                    raise TurnCancelled(payload['reason'])
                if payload['kind']=='fatal':raise RuntimeError(payload['reason'])
                window,pcm,metadata=payload['value']
                self.source_eof=metadata['source_eof_received']
                self.source_samples=metadata['input_samples_received']
                runtime.last_audio=metadata
                return window,pcm,metadata
            def close(self):
                if rank_source is not None:rank_source.close()
        self.source=DistributedSource()
        return self.source

    def stop_before(self,block):
        payload=None
        if self.rank==0:
            turn=self.coordinator.current
            payload=dict(cancel=turn.queue.cancelled,reason=turn.queue.reason)
        payload=self.broadcast(payload)
        if payload['cancel']:
            self.cancelled=True;self.cancel_reason=payload['reason'];return True
        return block>=self.pipeline.expected_blocks

    def after_audio(self,block):
        from video_timing import target_frame_count,required_blocks
        self.audio_snapshots[block]=copy.deepcopy(self.last_audio)
        if self.source.source_eof:
            frames=target_frame_count(self.source.source_samples/16000,20)
            blocks=required_blocks(frames)
            if not 10<=frames<=300 or blocks<self.pipeline.submitted:
                raise ValueError('EOS inconsistent with submitted output or turn bounds')
            if self.pipeline.effective_frames>frames:
                raise ValueError('Already emitted frames exceed actual EOS')
            self.pipeline.target_frames=frames;self.pipeline.expected_blocks=blocks
        return self.pipeline.target_frames,self.pipeline.expected_blocks

    def publish(self,tensor,row):
        snapshot=self.audio_snapshots.pop(row['block'])
        if self.rank==0:
            self.writer.offer(tensor,dict(row,input_snapshot=snapshot))

    def finish(self,index):
        from scripts.hopper_resident_transport import write
        if self.source is not None:self.source.close()
        if self.cancelled and self.rank==0:
            self.coordinator.current.cancel(self.cancel_reason or 'Renderer cancelled turn')
        if self.writer is not None:self.writer.close()
        pipeline=self.pipeline
        report=dict(status='cancelled' if self.cancelled else 'completed',turn_id=self.turn['id'],
            rank=self.rank,world_size=2,task_index=index,model_loads=self.model_loads,
            cleared_kv_tensors=self.cleared,seed_reset_in_original_generator=True,
            condition_cache_hits={key:self.cache_hits[key]-self.hits_before[key] for key in self.cache_hits},
            condition_cache_misses={key:self.cache_misses[key]-self.misses_before[key] for key in self.cache_misses},
            elapsed_seconds=time.monotonic()-self.turn_start,cancel_reason=self.cancel_reason,
            pipeline=dict(**pipeline.report(),backend=pipeline.backend.report()),
            input_eos_seen=bool(self.source.source_eof),input_samples_consumed=self.source.source_samples,
            last_audio=self.last_audio,output_queue_capacity=2,
            output_queue_peak=self.writer.peak_pending if self.writer else None,
            gpu_allocated_bytes=self.generator.torch.cuda.memory_allocated(),
            gpu_reserved_bytes=self.generator.torch.cuda.memory_reserved(),
            actual_target_frames=pipeline.target_frames,declared_total_samples=None,
            raw_audio_loaded_by_generator=False,real_time_accepted=False)
        write(Path.cwd()/f'resident-task-rank-{self.rank}.json',report)
        pipeline.close()
        # A short barrier is also the handover between old and new turn IDs.
        self.generator.dist.barrier(group=self.group)
        if self.rank==0:self.coordinator.finish_turn(report)
        self.tasks.append(dict(id=self.turn['id'],status=report['status']))
        self.tasks=self.tasks[-16:]
        self.source=None;self.writer=None;self.pipeline=None
        os.chdir(self.root)

    def close(self):
        if self.source is not None:self.source.close()
        if self.writer is not None:self.writer.close()
        if self.pipeline is not None:self.pipeline.close()
        if self.coordinator is not None:self.coordinator.close()
        os.chdir(self.root)


def resident_overlap_main():
    from scripts import hopper_runtime_cache,hopper_generate
    if os.environ.get('LIVEACT_HOPPER_VAE_MODE')!='overlap':
        raise ValueError('Use the accepted two-slot VAE overlap')
    if any(os.environ.get(key) for key in ('LIVEACT_P7_TIMELINE','LIVEACT_P10_PROFILE_ROOT',
        'LIVEACT_P10_REPRO_ROOT','LIVEACT_HOPPER_FP8_AUDIT')):
        raise ValueError('Resident entry does not enable profiling/audit backends')
    original_smoke=hopper_runtime_cache.pointwise_cache_smoke
    adapters=[];runtime=None;restore=None;original_trace=None
    status='failed'

    def initialized(rank,directory):
        nonlocal runtime,restore,original_trace
        result=original_smoke(rank,directory)
        import generate as generator
        from model_liveact.bounded_video_pipeline import BoundedVideoPipeline
        from model_liveact.vae_overlap import CUDAOverlapBackend,GeneratorOverlapAdapter
        from scripts.hopper_resident_adapter import install
        from src.audio_analysis.open_pcm import TurnCancelled
        template=json.loads(Path(os.environ['LIVEACT_H6_TEMPLATE']).read_text())[0]
        runtime=Runtime(generator,rank,Path.cwd(),template)

        class PublishingPipeline(BoundedVideoPipeline):
            def _complete(self,outputs):
                super()._complete(outputs)
                runtime.publish(outputs[-1],self.completed[-1])

        def create(vae,gpu_rank,target_frames,blocks):
            if target_frames!=300 or blocks!=10:
                raise ValueError('Open turn must start with the 15-second upper budget')
            pipeline=PublishingPipeline(CUDAOverlapBackend(vae,rank=gpu_rank,world_size=2),
                mode='overlap',target_frames=target_frames,expected_blocks=blocks)
            runtime.pipeline=pipeline
            return pipeline

        adapter=GeneratorOverlapAdapter(generator,create);adapter.install();adapters.append(adapter)
        hooks=dict(_h6_tasks=runtime.cases,_h6_start=runtime.start,
            _h6_text=lambda builder:runtime.cached('text',builder),
            _h6_image=lambda image,builder:runtime.cached('image',builder),
            _h6_vae=lambda image,builder:runtime.cached('vae',builder),
            _h6_build_image=runtime.build_image,_h6_max_duration=lambda:15,
            _h6_audio_source=runtime.audio,_h6_stop_before_block=runtime.stop_before,
            _h6_after_audio=runtime.after_audio,_h6_cancelled=lambda:runtime.cancelled,
            _h6_mark_cancelled=lambda:setattr(runtime,'cancelled',True),
            _h6_finish=runtime.finish,_H6TurnCancelled=TurnCancelled)
        restore=install(generator,adapter,hooks)
        original_trace=generator._p7_trace
        def trace(stage,**details):
            runtime.trace(stage,**details)
            return original_trace(stage,**details)
        generator._p7_trace=trace
        return result

    hopper_runtime_cache.pointwise_cache_smoke=initialized
    try:
        hopper_generate.main();status='completed'
    finally:
        root=Path(os.environ['LIVEACT_H6_ROOT']).resolve()
        if runtime is not None:
            report=dict(status=status,rank=int(os.environ['RANK']),world_size=2,
                model_loads=runtime.model_loads,model_ready_count=runtime.model_ready_count,
                recent_tasks=runtime.tasks,condition_cache_hits=runtime.cache_hits,
                condition_cache_misses=runtime.cache_misses,condition_cache_entries=len(runtime.cache),
                real_time_accepted=False)
            runtime.close()
            (root/f'resident-rank-{os.environ["RANK"]}.json').write_text(json.dumps(report,indent=2)+'\n')
            runtime.generator._p7_trace=original_trace
        if restore is not None:restore()
        for adapter in adapters:adapter.close()
        hopper_runtime_cache.pointwise_cache_smoke=original_smoke


def main():
    from scripts import hopper_vae_overlap,hopper_norm_generate
    if os.environ.get('LIVEACT_HOPPER_NORM_MODE')!='all':
        raise ValueError('Use the exact accepted C2 normalization backend')
    original=hopper_vae_overlap.main;hopper_vae_overlap.main=resident_overlap_main
    try:hopper_norm_generate.main()
    finally:hopper_vae_overlap.main=original


if __name__=='__main__':main()
