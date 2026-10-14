"""Checked AST hooks for resident SP4 experiments, retaining the original math.

The original generator already loads weights once outside its condition loop.
This adapter supplies live packets instead of a preloaded future waveform,
clears KV state on each condition and publishes decoded blocks immediately.
"""
import ast
import inspect
import textwrap


def transform_source(source):
    tree = ast.parse(textwrap.dedent(source))
    counts = dict(reset=0, metadata=0, duration=0, packets=0, publish=0, finish=0)

    class Hooks(ast.NodeTransformer):
        def visit_Expr(self, node):
            expression = ast.unparse(node.value)
            if expression == "_p7_trace('condition_start')":
                counts['reset'] += 1
                return [ast.parse('_r3_start(case_index, data, kv_cache)').body[0], node]
            if expression == 'gen_video_list.append(_videos.cpu())':
                counts['publish'] += 1
                return [node, ast.parse('_r3_publish(gen_video_list[-1], case_index, _, target_frames, fps)').body[0]]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            text = ast.unparse(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [ast.unparse(t) for t in node.targets[0].elts] == ['audio_ori', 'sr_ori']
                    and ast.unparse(node.value) == 'torchaudio.load(audio_path)'):
                counts['metadata'] += 1
                return ast.parse('audio_len = _r3_audio_duration(audio_path)').body[0]
            if text == 'audio_len = audio_ori.size(1) / sr_ori':
                counts['duration'] += 1
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == 'args.audio_stream_pcm':
                # Match the actual four setup statements, then preserve the
                # accepted conditioner/embedding construction verbatim.
                if (len(node.body) != 5 or not isinstance(node.body[0], ast.If)
                        or not ast.unparse(node.body[1]).startswith('integer_pcm = ')
                        or not isinstance(node.body[2], ast.If)
                        or not ast.unparse(node.body[3]).startswith('streaming_pcm = ReplayPreparedPCM(')
                        or not ast.unparse(node.body[4]).startswith('bounded_audio = StreamingAudioConditioner(')):
                    raise ValueError('Unknown original PCM construction')
                counts['packets'] += 1
                # This entry only permits streaming PCM; discard the offline
                # alternative rather than leaving undefined audio_ori uses.
                return [ast.parse('streaming_pcm = _r3_audio_source(audio_path, fps)').body[0], node.body[4]]
            return self.generic_visit(node)

        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node, ast.For) and ast.unparse(node.iter) == 'enumerate(input_data)':
                counts['finish'] += 1
                node.body.append(ast.parse('_r3_finish(case_index)').body[0])
            return node

    tree = ast.fix_missing_locations(Hooks().visit(tree))
    if counts != dict(reset=1, metadata=1, duration=1, packets=1, publish=1, finish=1):
        raise ValueError(f'Unknown resident generator structure: {counts}')
    return tree


def adapt_generator(original, hooks):
    required = {'_r3_start', '_r3_audio_duration', '_r3_audio_source', '_r3_publish', '_r3_finish'}
    if set(hooks) != required:
        raise ValueError('Complete resident hooks required')
    tree = transform_source(inspect.getsource(original))
    namespace = dict(original.__globals__, **hooks)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    return namespace[original.__name__]


def reset_kv(cache):
    """Reset all three timestep caches in place; never share task history."""
    tensors = 0
    if any(entry.get('fp8_kv_cache') or entry.get('offload_cache')
           for layers in cache.values() for entry in layers.values()):
        raise ValueError('Resident SP4 only accepts GPU BF16 KV')
    for layers in cache.values():
        for entry in layers.values():
            for key in ('k', 'v'):
                entry[key].zero_()
                tensors += 1
    return tensors
