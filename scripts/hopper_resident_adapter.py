"""Compose resident lifecycle hooks with the accepted output adapter's AST."""
import ast
import copy
from functools import wraps


def transform_tree(original):
    tree = copy.deepcopy(original)
    counts = dict(tasks=0,start=0,text=0,image=0,vae=0,audio=0,duration=0,
                  packets=0,stop=0,seal=0,flush=0,export=0,finish=0)

    class Hooks(ast.NodeTransformer):
        def visit_With(self, node):
            if (len(node.items)==1 and ast.unparse(node.items[0].context_expr)=="open(args.input_json, 'r', encoding='utf-8')"):
                if len(node.body)!=1 or ast.unparse(node.body[0])!='input_data = json.load(f)':
                    raise ValueError('Unknown task-input reader')
                counts['tasks']+=1
                return ast.parse('input_data = _h6_tasks(args.input_json)').body[0]
            return self.generic_visit(node)

        def visit_Expr(self,node):
            text=ast.unparse(node.value)
            if text=="_p7_trace('condition_start')":
                counts['start']+=1
                return [ast.parse('_h6_start(case_index, data, kv_cache)').body[0],
                        ast.parse('pre_latent = None').body[0],node]
            if text=='_r2_pipeline.flush(gen_video_list)':
                counts['flush']+=1
                return ast.parse('if not _h6_cancelled():\n    _r2_pipeline.flush(gen_video_list)\nelse:\n    _r2_pipeline.close()').body[0]
            return self.generic_visit(node)

        def visit_Assign(self,node):
            text=ast.unparse(node)
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple)
                    and [ast.unparse(n) for n in node.targets[0].elts]==['audio_ori','sr_ori']
                    and ast.unparse(node.value)=='torchaudio.load(audio_path)'):
                counts['audio']+=1
                return ast.parse('audio_len = _h6_max_duration()').body[0]
            if text=='audio_len = audio_ori.size(1) / sr_ori':
                counts['duration']+=1
                return None
            if text=='y = get_y(frame_num)':
                counts['vae']+=1
                return ast.parse('y = _h6_vae(image_path, lambda: get_y(frame_num))').body[0]
            return self.generic_visit(node)

        def visit_If(self,node):
            text=ast.unparse(node.test)
            if text=='args.audio_stream_pcm':
                if len(node.body)!=5 or not ast.unparse(node.body[-1]).startswith('bounded_audio = StreamingAudioConditioner('):
                    raise ValueError('Unknown accepted audio conditioner construction')
                counts['packets']+=1
                return [ast.parse('streaming_pcm = _h6_audio_source(audio_path, fps)').body[0],node.body[-1]]
            if text=='bounded_audio is not None':
                counts['seal']+=1
                updated=ast.parse('try:\n    pass\nexcept _H6TurnCancelled:\n    _h6_mark_cancelled()\n    break').body[0]
                updated.body=[node,ast.parse('target_frames, iter_total_num = _h6_after_audio(_)').body[0]]
                return updated
            return self.generic_visit(node)

        def visit_For(self,node):
            node=self.generic_visit(node)
            iterator=ast.unparse(node.iter)
            if iterator=='range(iter_total_num)' and ast.unparse(node.target)=='_':
                counts['stop']+=1
                node.body.insert(0,ast.parse('if _h6_stop_before_block(_):\n    break').body[0])
            if iterator=='enumerate(input_data)':
                # T5 returns immutable conditions; cache the actual expressions.
                body=node.body
                context=next((i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                              and ast.unparse(n.targets[0])=='context'),None)
                if context is None or ast.unparse(body[context+1].targets[0])!='edit_prompts':
                    raise ValueError('Unknown T5 condition sequence')
                build=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                                 body=ast.Tuple(elts=[body[context].value,body[context+1].value],ctx=ast.Load()))
                body[context:context+2]=[ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id='context',ctx=ast.Store()),ast.Name(id='edit_prompts',ctx=ast.Store())],ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id='_h6_text',ctx=ast.Load()),args=[build],keywords=[]))]
                counts['text']+=1
                image=next((i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                            and ast.unparse(n.targets[0])=='image'),None)
                if image is None or [ast.unparse(n) for n in body[image+2:image+6]]!=[
                    'clip.model.to(device)','clip_context = clip.visual(cond_image)','clip.model.cpu()','torch_gc()']:
                    raise ValueError('Unknown accepted image/CLIP sequence')
                body[image:image+6]=ast.parse('cond_image, clip_context = _h6_image(image_path, lambda: _h6_build_image(Image, transform, image_path, clip, device, torch))').body
                counts['image']+=1
                export=next((n for n in reversed(body) if isinstance(n,ast.If) and ast.unparse(n.test)=='rank == 0'),None)
                if export is None or not any('export_to_video' in ast.unparse(n) for n in export.body):
                    raise ValueError('Unknown final video export')
                export.test=ast.parse('rank == 0 and not _h6_cancelled()',mode='eval').body
                counts['export']+=1
                body.append(ast.parse('_h6_finish(case_index)').body[0]);counts['finish']+=1
            return node

    tree=ast.fix_missing_locations(Hooks().visit(tree))
    if any(value!=1 for value in counts.values()):
        raise ValueError('Unknown resident/overlap generator structure: '+str(counts))
    return tree


def install(generator, overlap_adapter, hooks):
    tree=transform_tree(overlap_adapter.transformed_tree)
    namespace=generator.__dict__
    previous={key:(key in namespace,namespace.get(key)) for key in hooks}
    namespace.update(hooks)
    original=generator.generate
    exec(compile(tree,str(generator.__file__),'exec'),namespace)
    generator.generate=wraps(overlap_adapter.original)(namespace[tree.body[0].name])
    def close():
        generator.generate=original
        for key,(existed,value) in previous.items():
            if existed:namespace[key]=value
            else:namespace.pop(key,None)
    return close
