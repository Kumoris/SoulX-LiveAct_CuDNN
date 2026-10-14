"""Unequal SP8 shards with unpadded head/sequence all-to-all transport.

Retain the accepted two virtual history/compression owners. Padding is allowed
only inside cache/head all-gather transport and is removed before any attention.
"""
import ast
from functools import wraps
import inspect
import math
import textwrap

import torch
import torch.distributed as dist
from torch.nn import functional as F

from model_liveact.parallel_geometry import ParallelGeometrySession


def shard_sizes(total, count):
    width=(total+count-1)//count
    sizes=tuple(min(width,max(0,total-i*width)) for i in range(count))
    if min(sizes)<=0:raise ValueError('Empty shards are not supported')
    return sizes


class VariableGeometry:
    def __init__(self,frame_tokens,world,rank):
        if world not in (2,4,8) or not 0<=rank<world or frame_tokens<=0:
            raise ValueError('Unsupported sequence-parallel geometry')
        self.frame_tokens,self.world,self.rank=frame_tokens,world,rank
        self.virtual_size=world//2;self.virtual_rank=rank//self.virtual_size;self.member=rank%self.virtual_size
        self.histories=shard_sizes(3*frame_tokens,self.virtual_size)
        self.currents=shard_sizes(4*frame_tokens,self.virtual_size)
        self.capacities=tuple(a+b for a,b in zip(self.histories,self.currents))
        self.history=self.histories[self.member];self.current=self.currents[self.member]

    def query_sizes(self,frames):
        if frames not in (6,8):raise ValueError('Only the fixed six/eight latent-frame blocks are supported')
        return (self.histories if frames==6 else self.currents)*2

    def query_offset(self,frames):
        return sum(self.query_sizes(frames)[:self.rank])

    def key_sizes(self,first):
        return (self.histories if first else self.capacities)*2

    def key_ids(self,first,device):
        hist=self.virtual_rank*3*self.frame_tokens+sum(self.histories[:self.member])
        ids=torch.arange(hist,hist+self.history,device=device)
        if first:return ids
        current=6*self.frame_tokens+self.virtual_rank*4*self.frame_tokens+sum(self.currents[:self.member])
        return torch.cat((ids,torch.arange(current,current+self.current,device=device)))

    def history_slice(self,virtual):
        start=sum(self.histories[:self.member]);return virtual[:,start:start+self.history]

    def current_slice(self,virtual):
        start=3*self.frame_tokens+sum(self.currents[:self.member]);return virtual[:,start:start+self.current]

    def reconstruct_virtual(self,parts):
        if len(parts)!=self.virtual_size or any(p.shape[1]!=c for p,c in zip(parts,self.capacities)):
            raise ValueError('Incomplete unequal virtual cache gather')
        return torch.cat([p[:,:h] for p,h in zip(parts,self.histories)]+
                         [p[:,h:] for p,h in zip(parts,self.histories)],dim=1)

    def audio_segments(self,frames):
        offset=self.query_offset(frames);stop=offset+self.query_sizes(frames)[self.rank]
        result=[];position=offset
        while position<stop:
            frame=position//self.frame_tokens;end=min(stop,(frame+1)*self.frame_tokens)
            result.append((position-offset,end-offset,frame));position=end
        return result


def pack_heads(x,world):
    b,s,h,d=x.shape
    if h%world:raise ValueError('Attention heads must divide the parallel world')
    hp=h//world
    return x.reshape(b,s,world,hp,d).permute(2,0,1,3,4).contiguous().view(-1),[b*s*hp*d]*world


def unpack_heads(flat,lengths,b,hp,d):
    sizes=[b*s*hp*d for s in lengths]
    return torch.cat([p.reshape(b,s,hp,d) for p,s in zip(flat.split(sizes),lengths)],dim=1)


def pack_sequence(x,lengths):
    b,s,hp,d=x.shape
    if s!=sum(lengths):raise ValueError('Head attention output has an invalid total sequence')
    return torch.cat([p.contiguous().view(-1) for p in x.split(lengths,dim=1)]),[b*n*hp*d for n in lengths]


def unpack_sequence(flat,world,b,local_s,hp,d):
    return torch.cat([p.reshape(b,local_s,hp,d) for p in flat.split(b*local_s*hp*d)],dim=2)


class VariableParallelSession(ParallelGeometrySession):
    def __init__(self,module,*,gather=None,fused_rotary=None):
        super().__init__(module,gather=gather,fused_rotary=fused_rotary,query_rotary=self.query_rope)
        self.active_geometry=None;self.active_first=None
        self.counts.update(query_rotary_calls=0,variable_attention_calls=0,head_all_to_all_calls=0,
            model_variable_shards=0,model_head_gathers=0,first_frame_suppressed_tokens=0)

    def initialize_groups(self,world):
        if self.gather is not None or world==2 or self.groups:return
        if not dist.is_initialized() or dist.get_world_size()!=world:
            raise ValueError('One complete same-node SP world required')
        width=world//2
        for virtual in range(2):self.groups[virtual]=dist.new_group(ranks=list(range(virtual*width,(virtual+1)*width)))

    def collect(self,tensor,g):
        if self.gather is not None:return self.gather(tensor,g)
        self.initialize_groups(g.world)
        maximum=max(g.capacities)
        padded=F.pad(tensor,(0,0,0,0,0,maximum-tensor.shape[1])).contiguous()
        result=[torch.empty_like(padded) for _ in range(g.virtual_size)]
        if tensor.is_cuda:torch.cuda.current_stream(tensor.device).synchronize()
        dist.all_gather(result,padded,group=self.groups[g.virtual_rank])
        if tensor.is_cuda:torch.cuda.current_stream(tensor.device).synchronize()
        self.counts['subgroup_gathers']+=1
        return [p[:,:n] for p,n in zip(result,g.capacities)]

    def prepare_cache(self,instance,k_cache,v_cache,k,v,kv_cache,grid_sizes,
                      sp_size,sp_rank,start_idx,end_idx,update_cache):
        frame=math.prod(grid_sizes[0][1:]).item();g=VariableGeometry(frame,sp_size,sp_rank);first=start_idx==0
        self.active_geometry,self.active_first=g,first
        if (end_idx!=(6 if first else 14)*frame or k_cache.shape[1]!=g.history+g.current
                or k.shape[1]!=(g.history if first else g.current)
                or kv_cache.get('offload_cache') or kv_cache.get('fp8_kv_cache')):
            raise ValueError('Unsupported variable cache/block/precision contract')
        if update_cache:
            if first:raise ValueError('First block cannot compress history')
            if sp_size==2:self.compress(instance,k_cache,v_cache,kv_cache,sp_rank,frame)
            else:
                parts=self.collect(torch.cat((k_cache,v_cache),dim=0),g)
                virtual=g.reconstruct_virtual(parts);vk,vv=virtual.chunk(2,dim=0)
                self.compress(instance,vk,vv,kv_cache,g.virtual_rank,frame)
                k_cache[:,:g.history].copy_(g.history_slice(vk));v_cache[:,:g.history].copy_(g.history_slice(vv))
            self.counts['compression_calls']+=1
        destination=slice(0,g.history) if first else slice(g.history,None)
        k_cache[:,destination]=k;v_cache[:,destination]=v
        length=g.history if first else g.history+g.current
        identity=(frame,sp_size,sp_rank,first,k.device,'key')
        if identity not in self.indices:self.indices[identity]=g.key_ids(first,k.device)
        self.counts['cache_calls']+=1
        return k_cache[:,:length],v_cache[:,:length],(identity,self.indices[identity])

    def query_rope(self,x,grid_sizes,freqs,sp_size,sp_rank,start_frame=0,_f=None):
        frames,h,w=grid_sizes.tolist()[0];frames=_f if _f else frames
        g=VariableGeometry(h*w,sp_size,sp_rank)
        if x.shape[1]!=g.query_sizes(frames)[sp_rank] or start_frame!=(0 if frames==6 else 6):
            raise ValueError('Query ownership/rotary start mismatch')
        identity=(h*w,sp_size,sp_rank,frames==6,x.device,'query')
        if identity not in self.indices:
            start=start_frame*h*w+g.query_offset(frames)
            self.indices[identity]=torch.arange(start,start+x.shape[1],device=x.device)
        self.counts['query_rotary_calls']+=1
        result=self.key_rope(x,grid_sizes,freqs,(identity,self.indices[identity]))
        self.counts['key_rotary_calls']-=1
        return result

    def audio(self,instance,x,encoder_hidden_states,sp_size,sp_rank,shape=None,start_f=0):
        frames,h,w=[int(v) for v in shape];g=VariableGeometry(h*w,sp_size,sp_rank)
        if x.shape[0]!=1 or x.shape[1]!=g.query_sizes(frames)[sp_rank]:raise ValueError('Audio query shard mismatch')
        states=encoder_hidden_states.squeeze(0)
        q=instance.q_linear(x).view(1,x.shape[1],instance.num_heads,instance.head_dim).permute(0,2,1,3)
        if instance.qk_norm:q=instance.q_norm(q)
        kv=instance.kv_linear(states).view(states.shape[0],states.shape[1],2,instance.num_heads,instance.head_dim)
        outputs=[]
        for begin,end,frame in g.audio_segments(frames):
            index=start_f+frame
            if index>=kv.shape[0]:raise ValueError('Audio condition outside temporal range')
            key,value=kv[index:index+1].permute(2,0,3,1,4).unbind(0)
            if instance.qk_norm:key=instance.add_k_norm(key)
            result=F.scaled_dot_product_attention(q[:,:,begin:end],key,value,dropout_p=0.0,is_causal=False)
            outputs.append(result.transpose(1,2).reshape(1,end-begin,instance.dim))
            if end-begin!=h*w:self.counts['partial_audio_segments']+=1
        result=instance.proj_drop(instance.proj(torch.cat(outputs,dim=1)))
        # Original WanAttentionBlock zeros reference-frame audio on rank0.
        # For SP8 that frame spans rank0 and rank1; preserve its global extent.
        if start_f==0:
            count=max(0,min(result.shape[1],h*w-g.query_offset(frames)))
            if count:result[:,:count]=0;self.counts['first_frame_suppressed_tokens']+=count
        self.counts['audio_calls']+=1
        return result

    def _exchange(self,sent,input_sizes,output_sizes):
        received=torch.empty(sum(output_sizes),dtype=sent.dtype,device=sent.device)
        dist.all_to_all_single(received,sent,output_split_sizes=output_sizes,input_split_sizes=input_sizes)
        self.counts['head_all_to_all_calls']+=1
        return received

    def head_exchange(self,x,lengths):
        world=len(lengths);b,s,h,d=x.shape;rank=dist.get_rank()
        if s!=lengths[rank] or h%world:raise ValueError('Unequal attention ownership mismatch')
        sent,ins=pack_heads(x,world);hp=h//world;outs=[b*n*hp*d for n in lengths]
        return unpack_heads(self._exchange(sent,ins,outs),lengths,b,hp,d)

    def sequence_exchange(self,x,lengths):
        world=len(lengths);b,_,hp,d=x.shape;local=lengths[dist.get_rank()]
        sent,ins=pack_sequence(x,lengths);outs=[b*local*hp*d]*world
        return unpack_sequence(self._exchange(sent,ins,outs),world,b,local,hp,d)

    def attention_factory(self,**kwargs):
        def attention(_unused,*,query,key,value,window_size):
            if tuple(window_size)!=(-1,-1):raise ValueError('Only the original unrestricted attention window is supported')
            from yunchang.kernels.attention import pytorch_attn_forward
            g=self.active_geometry;frames=6 if self.active_first else 8
            lengths=g.query_sizes(frames);keys=g.key_sizes(self.active_first)
            q=self.head_exchange(query,lengths)
            packed=self.head_exchange(torch.cat((key,value),dim=0),keys);k,v=packed.chunk(2,dim=0)
            output=pytorch_attn_forward(q,k,v,causal=False,op_type='flash')[0]
            self.counts['variable_attention_calls']+=1
            return self.sequence_exchange(output,lengths)
        return attention

    def transformed_forward(self):
        forward=super().transformed_forward()
        forward.__globals__['xFuserLongContextAttention']=self.attention_factory
        return forward

    def shard_model_tokens(self,x,grid_sizes):
        frames,h,w=grid_sizes.tolist()[0];world=self.module.get_sequence_parallel_world_size();rank=self.module.get_sequence_parallel_rank()
        g=VariableGeometry(h*w,world,rank)
        if x.shape[1]!=frames*h*w:raise ValueError('Unpatchified token count mismatch')
        self.counts['model_variable_shards']+=1
        return x.split(g.query_sizes(frames),dim=1)[rank]

    def gather_model_tokens(self,x,grid_sizes):
        frames,h,w=grid_sizes.tolist()[0];world=self.module.get_sequence_parallel_world_size();rank=self.module.get_sequence_parallel_rank()
        sizes=VariableGeometry(h*w,world,rank).query_sizes(frames);maximum=max(sizes)
        if x.shape[1]!=sizes[rank]:raise ValueError('Model head output ownership mismatch')
        padded=F.pad(x,(0,0,0,maximum-x.shape[1])).contiguous()
        result=[torch.empty_like(padded) for _ in range(world)]
        if x.is_cuda:torch.cuda.current_stream(x.device).synchronize()
        dist.all_gather(result,padded)
        if x.is_cuda:torch.cuda.current_stream(x.device).synchronize()
        self.counts['model_head_gathers']+=1
        return torch.cat([p[:,:n] for p,n in zip(result,sizes)],dim=1)

    def install(self):
        cls=self.module.WanSelfAttention;original_init=cls.init_kvidx;forward=self.transformed_forward();session=self
        def initialize(instance,frame_len,world):
            g=VariableGeometry(frame_len,world,session.module.get_sequence_parallel_rank())
            session.initialize_groups(world)
            instance.kv_idx0=torch.arange(g.history,device=instance.q.weight.device)
            instance.kv_idx2=torch.arange(g.history+g.current,device=instance.q.weight.device)
        cls.forward,cls.init_kvidx=forward,initialize
        self.restores += [(cls,'forward',self.original_forward),(cls,'init_kvidx',original_init)]
        audio=self.module.SingleStreamAttention;original_audio=audio.forward
        def audio_forward(instance,*args,**kwargs):return session.audio(instance,*args,**kwargs)
        audio.forward=audio_forward;self.restores.append((audio,'forward',original_audio))
        if hasattr(self.module,'WanModel'):
            original=self.module.WanModel.forward
            tree=ast.parse(textwrap.dedent(inspect.getsource(original)));found={'shard':0,'gather':0}
            for node in tree.body[0].body:
                if not isinstance(node,ast.Assign) or ast.unparse(node.targets[0])!='x':continue
                value=ast.unparse(node.value)
                if value=='torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]':
                    node.value=ast.parse('_variable_shard(x,grid_sizes)',mode='eval').body;found['shard']+=1
                elif value=='get_sp_group().all_gather(x, dim=1)':
                    node.value=ast.parse('_variable_gather(x,grid_sizes)',mode='eval').body;found['gather']+=1
            if found!={'shard':1,'gather':1}:raise ValueError('Unknown original model shard/gather source')
            ns=dict(original.__globals__);ns.update(_variable_shard=self.shard_model_tokens,_variable_gather=self.gather_model_tokens)
            exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(original),'exec'),ns)
            self.module.WanModel.forward=wraps(original)(ns['forward']);self.restores.append((self.module.WanModel,'forward',original))

    def patch_generator(self,generator):
        original=generator.generate;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));found=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='kv_cache_tokens':
                if ast.unparse(node.value)!='frame_len * sum(blksz_lst) // world_size':raise ValueError('Unknown generator cache shape')
                node.value=ast.parse('_variable_cache_tokens(frame_len,world_size,rank)',mode='eval').body;found+=1
        if found!=1:raise ValueError('One original cache allocation required')
        tree.body[0].name='_p10_variable_generate';ns=original.__globals__
        if '_variable_cache_tokens' in ns or '_p10_variable_generate' in ns:raise ValueError('Adapter namespace already installed')
        ns['_variable_cache_tokens']=lambda frame,world,rank:sum((VariableGeometry(frame,world,rank).history,VariableGeometry(frame,world,rank).current))
        exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(original),'exec'),ns)
        generator.generate=wraps(original)(ns['_p10_variable_generate'])
        self.restores.append((generator,'generate',original));self.generator_namespace=ns

    def report(self):
        return dict(**self.counts,supported_world_sizes=[2,4,8],sp8_supported=True,
            temporal_compression_changed=False,weights_changed=False,attention_padding=False,
            communication_padding_trimmed=True,transport='variable-size all_to_all_single',
            resident_tables=len(self.tables),subgroup_count=len(self.groups))

    def close(self):
        super().close()
        if hasattr(self,'generator_namespace'):
            self.generator_namespace.pop('_variable_cache_tokens',None)
            self.generator_namespace.pop('_p10_variable_generate',None)
        self.active_geometry=None
