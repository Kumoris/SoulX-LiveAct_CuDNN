"""Experimental FP64 rotary math with PyTorch-compatible double rounding on SM80.

No lowered rotary precision: fuse conversion, FP64 complex multiplication and final
rounding. Actual GPU output must match the original BF16 attention inputs before
this can be tested in a generated video.
"""
import torch
import triton
import triton.language as tl

from model_liveact.positional_layout import RopeLayoutSession


@triton.jit
def _rotary_exact_kernel(X, FREQ, OUT, PAIRS:tl.constexpr, HEADS:tl.constexpr,
    HALF:tl.constexpr, STRIDE_S:tl.constexpr, STRIDE_H:tl.constexpr,
    STRIDE_D:tl.constexpr, BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=index<PAIRS
    channel=index%HALF
    head=(index//HALF)%HEADS
    token=index//(HALF*HEADS)
    offset=token*STRIDE_S+head*STRIDE_H+2*channel*STRIDE_D
    x0=tl.load(X+offset,mask=valid,other=0).to(tl.float64)
    x1=tl.load(X+offset+STRIDE_D,mask=valid,other=0).to(tl.float64)
    phase=2*(token*HALF+channel)
    cosine=tl.load(FREQ+phase,mask=valid,other=0)
    sine=tl.load(FREQ+phase+1,mask=valid,other=0)
    y0=x0*cosine-x1*sine
    y1=x0*sine+x1*cosine
    # PyTorch double -> BFloat16 converts via FP32. Triton's direct FP64 ->
    # BF16 lowering deliberately avoids double rounding and differs at ties.
    # Preserve both rounding stages explicitly without an intermediate tensor.
    y0_final=y0.to(tl.float32)
    y1_final=y1.to(tl.float32)
    tl.store(OUT+2*index,y0_final,mask=valid)
    tl.store(OUT+2*index+1,y1_final,mask=valid)


def exact_fused_rotary_final(x,table):
    if (x.device.type!='cuda' or x.dtype!=torch.bfloat16 or x.ndim!=4 or x.shape[0]!=1
            or x.shape[-1]%2 or table.dtype!=torch.complex128 or not table.is_contiguous()
            or table.shape!=(x.shape[1],1,x.shape[-1]//2) or table.device!=x.device):
        raise ValueError('Unsupported exact final rotary shape/type/layout')
    output=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    pairs=x.shape[1]*x.shape[2]*(x.shape[3]//2)
    _rotary_exact_kernel[(triton.cdiv(pairs,256),)](x,torch.view_as_real(table),output,
        pairs,x.shape[2],x.shape[3]//2,x.stride(1),x.stride(2),x.stride(3),256,
        num_warps=4,enable_fp_fusion=False)
    return output


class ExactFusedRopeSession(RopeLayoutSession):
    def __init__(self,module):
        super().__init__(module,'prefix-views')
        self.counts.update(fused_calls=0,fused_fallbacks=0)

    def _eligible(self,x,freqs):
        return (self._stamp(freqs) is not None and x.device.type=='cuda'
            and x.dtype==torch.bfloat16 and x.ndim==4 and x.shape[0]==1
            and x.shape[-1]%2==0 and freqs.dtype==torch.complex128)

    def causal(self,x,grid_sizes,freqs,sp_size,sp_rank,start_frame=0,_f=None):
        self.counts['causal_calls']+=1
        if not self._eligible(x,freqs) or len(grid_sizes)!=1:
            self.counts['fused_fallbacks']+=1
            return self.original_causal(x,grid_sizes,freqs,sp_size,sp_rank,start_frame,_f)
        s,c=x.size(1),x.size(3)//2
        f,h,w=grid_sizes.tolist()[0];f=_f if _f else f
        key=('query',c,f,h,w,s,sp_size,sp_rank,start_frame,x.device)
        table=self._table(freqs,key,lambda:self._make_table(freqs,c,f,h,w,start_frame,x.device)
            [sp_rank*s:(sp_rank+1)*s])
        self.counts['fused_calls']+=1
        return exact_fused_rotary_final(x,table)

    def rope(self,x,grid_sizes,freqs,f_list=(),rope_list=()):
        self.counts['key_calls']+=1
        pairs=tuple((tuple(f),tuple(r)) for f,r in zip(f_list,rope_list))
        _,h,w=grid_sizes.tolist()[0]
        supported=(bool(pairs) and pairs[0][0][0]==0 and pairs[-1][0][1]*h*w==x.shape[1]
            and all(end-start==r_end-r_start for (start,end),(r_start,r_end) in pairs)
            and all(pairs[i][0][1]==pairs[i+1][0][0] for i in range(len(pairs)-1)))
        if not self._eligible(x,freqs) or not supported:
            self.counts['fused_fallbacks']+=1
            return self.original_rope(x,grid_sizes,freqs,f_list,rope_list)
        c=x.size(3)//2;key=('key',c,h,w,pairs,x.device)
        table=self._table(freqs,key,lambda:torch.cat([
            self._make_table(freqs,c,end-start,h,w,r_start,x.device)
            for (start,end),(r_start,r_end) in pairs],dim=0))
        self.counts['fused_calls']+=1
        return exact_fused_rotary_final(x,table)

    def report(self):
        return {**super().report(),'mode':'fused-rotary','rotary_compute_dtype':'float64',
            'rotary_output_dtype':'final BF16 attention input','fp_fusion':False,'explicit_float32_before_bf16':True}
