"""Patch only two modulation expressions and two residual/cast pairs."""
import ast
from functools import wraps
import inspect
import textwrap

import torch

from model_liveact import pointwise_fused as operators


class PointwiseSession:
    def __init__(self,module):
        self.module=module
        self.original=module.WanAttentionBlock.forward
        self.counts=dict(modulation_calls=0,residual_calls=0,fused_modulation_calls=0,
            fused_residual_calls=0,modulation_fallbacks=0,residual_fallbacks=0)
        self.parameter_dtypes={};self.installed=False

    def _modulate(self,x,shift,scale):
        self.counts['modulation_calls']+=1
        key=str(scale.dtype);self.parameter_dtypes[key]=self.parameter_dtypes.get(key,0)+1
        eligible=operators._eligible(x,shift,scale)
        self.counts['fused_modulation_calls' if eligible else 'modulation_fallbacks']+=1
        return operators.modulation(x,shift,scale)

    def _residual(self,x,y,scale,dtype):
        self.counts['residual_calls']+=1
        eligible=(dtype==x.dtype and operators._eligible(x,scale) and y.shape==x.shape
            and y.dtype==x.dtype and y.device==x.device and y.is_contiguous())
        self.counts['fused_residual_calls' if eligible else 'residual_fallbacks']+=1
        return operators.residual(x,y,scale) if eligible else (x+y*scale).to(dtype)

    def install(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(self.original)))
        stamp=lambda node:ast.dump(node,include_attributes=False)
        expr=lambda text:ast.parse(text,mode='eval').body
        patterns={stamp(expr('(self.norm1(x).float()*(1+e[1])+e[0]).type_as(x)')):
                  '_r2_modulate(self.norm1(x),e[0],e[1])',
                  stamp(expr('(self.norm2(x).float()*(1+e[4])+e[3]).to(dtype)')):
                  '_r2_modulate(self.norm2(x),e[3],e[4])'}
        found={key:0 for key in patterns}
        class ReplaceModulation(ast.NodeTransformer):
            def visit_Call(self,node):
                key=stamp(node)
                if key in patterns:
                    found[key]+=1;return ast.copy_location(expr(patterns[key]),node)
                return self.generic_visit(node)
        tree=ReplaceModulation().visit(tree)
        if list(found.values())!=[1,1]:raise ValueError('Exactly two original modulation expressions required')
        body=tree.body[0].body;new=[];seen=[];i=0
        cast=stamp(ast.parse('x=x.to(dtype)').body[0])
        while i<len(body):
            hit=next((index for index in (2,5) if stamp(body[i])==stamp(ast.parse(f'x=x+y*e[{index}]').body[0])),None)
            if hit is not None:
                if i+1>=len(body) or stamp(body[i+1])!=cast:
                    raise ValueError('Residual must retain its original immediate dtype conversion')
                node=ast.parse(f'x=_r2_residual(x,y,e[{hit}],dtype)').body[0]
                new.append(ast.copy_location(node,body[i]));seen.append(hit);i+=2
            else:new.append(body[i]);i+=1
        if seen!=[2,5]:raise ValueError('Exactly two original residual/cast pairs required')
        tree.body[0].body=new;tree=ast.fix_missing_locations(tree)
        namespace=dict(self.original.__globals__)
        namespace.update(_r2_modulate=self._modulate,_r2_residual=self._residual)
        exec(compile(tree,inspect.getsourcefile(self.original),'exec'),namespace)
        self.module.WanAttentionBlock.forward=wraps(self.original)(namespace['forward'])
        self.installed=True

    def report(self):
        return {**self.counts,'parameter_dtypes':dict(self.parameter_dtypes),
            'normalization_changed':False,'gemm_changed':False,'weights_changed':False,
            'kv_changed':False,'fp_fusion':False,'bf16_intermediate_rounding_preserved':True}

    def close(self):
        if self.installed:self.module.WanAttentionBlock.forward=self.original
        self.installed=False
