"""Opt-in modulation/residual fusion preserving each eager BF16 rounding step.

Keep normalization and GEMMs unchanged. This module is only an operator probe;
no existing model forward path imports or enables it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _modulation(X, SHIFT, SCALE, OUT, COUNT:tl.constexpr, WIDTH:tl.constexpr,
                PARAM_BF16:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<COUNT
    value=tl.load(X+i,mask=mask,other=0).to(tl.float32)
    shift=tl.load(SHIFT+i%WIDTH,mask=mask,other=0).to(tl.float32)
    scale=1.0+tl.load(SCALE+i%WIDTH,mask=mask,other=0).to(tl.float32)
    if PARAM_BF16:scale=scale.to(tl.bfloat16).to(tl.float32)
    out=value*scale+shift
    tl.store(OUT+i,out,mask=mask)


@triton.jit
def _residual(X, Y, SCALE, OUT, COUNT:tl.constexpr, WIDTH:tl.constexpr,
              PARAM_BF16:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<COUNT
    value=tl.load(X+i,mask=mask,other=0).to(tl.float32)
    y=tl.load(Y+i,mask=mask,other=0).to(tl.float32)
    scale=tl.load(SCALE+i%WIDTH,mask=mask,other=0).to(tl.float32)
    product=y*scale
    if PARAM_BF16:product=product.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT+i,value+product,mask=mask)


def _eligible(x,*params):
    return (not torch.is_grad_enabled() and x.device.type=='cuda'
        and x.dtype==torch.bfloat16 and x.ndim==3 and x.shape[0]==1
        and x.is_contiguous() and all(p.device==x.device and p.is_contiguous()
            and p.shape==(1,1,x.shape[-1]) and p.dtype==params[0].dtype
            and p.dtype in (torch.bfloat16,torch.float32) for p in params))


def modulation(x,shift,scale):
    if not _eligible(x,shift,scale):
        return (x.float()*(1+scale)+shift).to(x.dtype)
    out=torch.empty_like(x)
    _modulation[(triton.cdiv(x.numel(),512),)](x,shift,scale,out,x.numel(),
        x.shape[-1],scale.dtype==torch.bfloat16,512,num_warps=4,enable_fp_fusion=False)
    return out


def residual(x,y,scale):
    if (not _eligible(x,scale) or y.shape!=x.shape or y.dtype!=x.dtype
            or y.device!=x.device or not y.is_contiguous()):
        return (x+y*scale).to(x.dtype)
    out=torch.empty_like(x)
    _residual[(triton.cdiv(x.numel(),512),)](x,y,scale,out,x.numel(),x.shape[-1],
        scale.dtype==torch.bfloat16,512,num_warps=4,enable_fp_fusion=False)
    return out
