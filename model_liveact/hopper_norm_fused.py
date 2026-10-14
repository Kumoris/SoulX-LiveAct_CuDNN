"""Experimental FP32 normalization reduction with explicit BF16 output stages.

Reduction order may differ from Torch. Kernel error gates and new full-video
human review are required before this numerical change can be adopted.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X,W,B,Y,C:tl.constexpr,EPS:tl.constexpr,RMS:tl.constexpr,
          AFFINE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK);valid=column<C
    value=tl.load(X+row*C+column,mask=valid,other=0).to(tl.float32)
    if RMS:
        variance=tl.sum(tl.where(valid,value*value,0),axis=0)/C
        normalized=value*tl.rsqrt(variance+EPS)
        # Actual WanRMSNorm casts the normalized value BEFORE multiplying the
        # BF16 weight. Preserve that intermediate rounding, not just final dtype.
        rounded=normalized.to(tl.bfloat16).to(tl.float32)
        weight=tl.load(W+column,mask=valid,other=0).to(tl.float32)
        output=rounded*weight
    else:
        mean=tl.sum(value,axis=0)/C
        centered=value-mean
        variance=tl.sum(tl.where(valid,centered*centered,0),axis=0)/C
        output=centered*tl.rsqrt(variance+EPS)
        if AFFINE:
            weight=tl.load(W+column,mask=valid,other=0).to(tl.float32)
            bias=tl.load(B+column,mask=valid,other=0).to(tl.float32)
            output=output*weight+bias
    tl.store(Y+row*C+column,output,mask=valid)


def fused_norm(value,weight,bias,eps,*,rms):
    if (not value.is_cuda or value.dtype!=torch.bfloat16 or not value.is_contiguous()
            or value.shape[-1]!=5120 or torch.is_grad_enabled()):
        raise ValueError('Only contiguous no-grad CUDA BF16 5120-wide normalization supported')
    for parameter in (weight,bias):
        if parameter is not None and (parameter.device!=value.device or parameter.dtype!=torch.bfloat16
                                     or not parameter.is_contiguous() or parameter.shape!=(5120,)):
            raise ValueError('Keep actual BF16 normalization parameter layout')
    if rms and weight is None:raise ValueError('RMS weight required')
    if not rms and ((weight is None)!=(bias is None)):raise ValueError('Affine weight/bias pair required')
    output=torch.empty_like(value)
    _norm[(value.numel()//5120,)](value,weight if weight is not None else value,
        bias if bias is not None else value,output,5120,eps,rms,weight is not None,8192,
        num_warps=8,enable_fp_fusion=False)
    return output


class NormSession:
    def __init__(self,module):
        self.module=module;self.originals={};self.calls={'rms':0,'layer':0};self.fallbacks={'rms':0,'layer':0}

    def install(self):
        for name,cls in (('rms',self.module.WanRMSNorm),('layer',self.module.WanLayerNorm)):
            original=cls.forward;self.originals[cls]=original
            def forward(instance,value,*,name=name,original=original):
                eligible=(value.is_cuda and value.dtype==torch.bfloat16 and value.is_contiguous()
                    and value.shape[-1]==5120 and not torch.is_grad_enabled()
                    and not instance.training)
                if not eligible:
                    self.fallbacks[name]+=1;return original(instance,value)
                self.calls[name]+=1
                return fused_norm(value,instance.weight,getattr(instance,'bias',None),instance.eps,rms=name=='rms')
            cls.forward=forward

    def report(self):
        return dict(calls=self.calls,fallbacks=self.fallbacks,compute_dtype='float32',output_dtype='bfloat16',
                    rms_intermediate_bf16_rounding_preserved=True,reduction_order_may_differ=True,
                    weights_steps_and_attention_changed=False,human_quality_accepted=False,real_time_accepted=False)

    def close(self):
        for cls,original in self.originals.items():cls.forward=original
        self.originals.clear()
