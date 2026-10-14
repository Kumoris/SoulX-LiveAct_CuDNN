"""Opt-in DiT BF16 cuDNN attention; real LSE, no global SDPA flag changes."""
import ast
import importlib
import inspect
import textwrap

import torch


def audio_forward_tree(source):
    tree=ast.parse(textwrap.dedent(source));count=0
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.unparse(node.func)=='torch.nn.functional.scaled_dot_product_attention':
                count+=1;node.func=ast.Name(id='_hopper_cudnn_sdpa',ctx=ast.Load())
            return self.generic_visit(node)
    tree=ast.fix_missing_locations(Replace().visit(tree))
    if count!=1 or len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Expected exactly one original audio attention call')
    tree.body[0].name='_hopper_cudnn_audio_forward'
    return tree


class NativeCudnnSession:
    def __init__(self,model):
        self.model=model
        self.ring=importlib.import_module('xfuser.core.long_ctx_attention.ring.ring_flash_attn')
        self.selector=self.ring.select_flash_attn_impl
        self.cross=model.sdpa_attention
        self.audio=model.SingleStreamAttention.forward
        self.calls=dict(self_attention=0,cross_attention=0,audio_attention=0)
        self.symbols={name:(name in model.__dict__,model.__dict__.get(name))
                      for name in ('_hopper_cudnn_sdpa','_hopper_cudnn_audio_forward')}
        self.installed=False

    def native(self,q,k,v,*,stats,dropout_p=0.,is_causal=False,scale=None,mask=None):
        if (mask is not None or dropout_p or is_causal or
                any(not value.is_cuda or value.dtype!=torch.bfloat16 for value in (q,k,v))):
            raise ValueError('Frozen Hopper cuDNN candidate requires unmasked noncausal CUDA BF16')
        output,lse=torch.ops.aten._scaled_dot_product_cudnn_attention.default(
            q,k,v,None,stats,0.,False,scale=scale)[:2]
        return output,lse

    def long_forward(self,q,k,v,dropout_p=0.,softmax_scale=None,causal=False,
                     window_size=(-1,-1),softcap=None,alibi_slopes=None,return_softmax=False):
        if window_size!=(-1,-1) or softcap not in (None,0.) or alibi_slopes is not None or return_softmax:
            raise ValueError('Unsupported long-context attention configuration')
        self.calls['self_attention']+=1
        output,lse=self.native(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                               stats=True,dropout_p=dropout_p,is_causal=causal,scale=softmax_scale)
        return output.transpose(1,2),lse

    def cross_forward(self,q,k,v,**options):
        if (options.get('q_lens') is not None or options.get('k_lens') is not None
                or options.get('q_scale') is not None or options.get('window_size',(-1,-1))!=(-1,-1)):
            raise ValueError('Unsupported cross-attention lengths or scaling')
        self.calls['cross_attention']+=1
        output,_=self.native(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),stats=False,
                              dropout_p=options.get('dropout_p',0.),is_causal=options.get('causal',False),
                              scale=options.get('softmax_scale'),mask=options.get('attn_mask'))
        return output.transpose(1,2).contiguous()

    def audio_sdpa(self,q,k,v,attn_mask=None,dropout_p=0.,is_causal=False):
        self.calls['audio_attention']+=1
        return self.native(q,k,v,stats=False,mask=attn_mask,
                           dropout_p=dropout_p,is_causal=is_causal)[0]

    def install(self):
        if self.installed:raise ValueError('Attention session already installed')
        from yunchang.kernels import AttnType
        def selector(kind,stage='fwd-only',attn_processor=None):
            if kind==AttnType.TORCH_FLASH:
                if stage!='fwd-only' or attn_processor is not None:
                    raise ValueError('Only the frozen forward-only Hopper attention may be replaced')
                return self.long_forward
            return self.selector(kind,stage=stage,attn_processor=attn_processor)
        tree=audio_forward_tree(inspect.getsource(self.audio))
        self.model.__dict__['_hopper_cudnn_sdpa']=self.audio_sdpa
        exec(compile(tree,inspect.getsourcefile(self.audio),'exec'),self.model.__dict__)
        self.model.SingleStreamAttention.forward=self.model.__dict__['_hopper_cudnn_audio_forward']
        self.model.sdpa_attention=self.cross_forward
        self.ring.select_flash_attn_impl=selector
        self.installed=True

    def report(self):
        return dict(operator='aten::_scaled_dot_product_cudnn_attention',dtype='torch.bfloat16',
                    calls=dict(self.calls),self_attention_real_lse=True,
                    global_sdpa_flags_changed=False,vae_attention_backend_changed=False,
                    whole_video_quality_accepted=False,real_time_accepted=False)

    def close(self):
        if self.installed:
            self.ring.select_flash_attn_impl=self.selector
            self.model.sdpa_attention=self.cross
            self.model.SingleStreamAttention.forward=self.audio
            for name,(existed,value) in self.symbols.items():
                if existed:self.model.__dict__[name]=value
                else:self.model.__dict__.pop(name,None)
        self.installed=False
