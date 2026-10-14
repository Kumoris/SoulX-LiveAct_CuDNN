"""Side-stream local VAE decode; all collectives stay on the owner thread."""
import ast
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import inspect
import textwrap
import threading

import torch


def split_decode(original, *, communication_wait):
    """Split the actual spatial decoder at its gather, with strict source guards."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    stamp = lambda node: ast.dump(node, include_attributes=False)
    marker = ast.parse('total_processes = world_size_h * world_size_w').body[0]
    found = [i for i, node in enumerate(function.body) if stamp(node) == stamp(marker)]
    if len(found) != 1 or function.name != 'decode_dist_2d':
        raise ValueError('Expected original spatial decoder gather boundary')
    index = found[0]
    gather = ast.parse('dist.all_gather(full_images, images_chunk)').body[0]
    wait = ast.parse('self.device_synchronize()').body[0]
    suffix = function.body[index:]
    if (sum(stamp(n) == stamp(gather) for n in suffix) != 1
            or sum(stamp(n) == stamp(wait) for n in suffix) != 1
            or any('dist.' in ast.unparse(n) or 'synchronize' in ast.unparse(n) for n in function.body[:index])):
        raise ValueError('Local decode must contain no communication or synchronization')
    local = ast.FunctionDef(name='_r2_local_decode', args=function.args,
        body=function.body[:index] + [ast.parse('return images_chunk').body[0]], decorator_list=[])
    join = ast.parse('def _r2_join_decode(self, images_chunk, world_size_h, world_size_w):\n    pass').body[0]
    join.body = [ast.parse('_r2_communication_wait(images_chunk)').body[0] if stamp(n) == stamp(wait) else n
                 for n in suffix]
    namespace = dict(original.__globals__)
    namespace['_r2_communication_wait'] = communication_wait
    compiled = ast.fix_missing_locations(ast.Module(body=[local, join], type_ignores=[]))
    exec(compile(compiled, inspect.getsourcefile(original), 'exec'), namespace)
    return namespace['_r2_local_decode'], namespace['_r2_join_decode']


class CUDAOverlapBackend:
    def __init__(self, vae, *, rank, world_size=2, timeout_seconds=120):
        if (not vae.parallel or not vae.use_2d_split or vae.use_tiling
                or vae.cpu_offload or world_size != 2):
            raise ValueError('Fixed resident SP2 spatial VAE required')
        self.vae = vae
        self.rank = rank
        self.world_size = world_size
        self.timeout = timeout_seconds
        self.owner = threading.get_ident()
        self.compute_stream = torch.cuda.current_stream(rank)
        self.decode_stream = torch.cuda.Stream(device=rank)
        self.copy_stream = torch.cuda.Stream(device=rank)
        # VAE weights were allocated/encoded on the compute stream.
        self.decode_stream.wait_stream(self.compute_stream)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='liveact-local-vae')
        self.local, self.join = split_decode(type(vae).decode_dist_2d,
            communication_wait=lambda tensor: torch.cuda.current_stream(tensor.device).synchronize())
        self.collective_blocks = []
        self.local_blocks = []
        self.pinned_buffers_peak = 0
        self.closed = False
        torch.cuda.reset_peak_memory_stats(rank)

    def _main(self):
        if threading.get_ident() != self.owner or self.closed:
            raise ValueError('VAE communication must stay on its live owner thread')

    def _local_work(self, block, latent, producer, grid):
        torch.cuda.set_device(self.rank)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), torch.cuda.stream(self.decode_stream):
            self.decode_stream.wait_event(producer)
            height, width = grid
            chunk = self.local(self.vae, latent, height, width, self.rank//width, self.rank%width)
            ready = torch.cuda.Event()
            ready.record(self.decode_stream)
            self.local_blocks.append(block)
        return dict(block=block, chunk=chunk, ready=ready, grid=grid, latent=latent)

    def submit(self, block, latent, *, asynchronous):
        self._main()
        if asynchronous:
            producer = torch.cuda.Event()
            producer.record(self.compute_stream)
            latent.record_stream(self.decode_stream)
            grid = self.vae.calculate_2d_grid(latent.shape[2], latent.shape[3], self.world_size)
            return self.executor.submit(self._local_work, block, latent, producer, grid)
        # Serial is the original complete-video decoder, including its sync.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.vae.decode(latent)
        return dict(block=block, serial_output=output)

    def finish(self, job):
        self._main()
        if isinstance(job, dict):
            output = job['serial_output']
            block = job['block']
            value = output[:, :, 9:] if block else output
            value = value.cpu()
            torch.cuda.synchronize(self.rank)
            return value
        partial = job.result(timeout=self.timeout)
        block = partial['block']
        # All previous SP collectives on the compute stream must finish before
        # switching to the VAE gather. No worker issues any collective.
        self.compute_stream.synchronize()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(partial['ready'])
            partial['chunk'].record_stream(self.compute_stream)
            output = self.join(self.vae, partial['chunk'], *partial['grid'])
            self.collective_blocks.append(block)
            output = output[:, :, 9:] if block else output
            ready = torch.cuda.Event()
            ready.record(self.compute_stream)
            output.record_stream(self.copy_stream)
        pinned = torch.empty(output.shape, dtype=output.dtype, device='cpu', pin_memory=True)
        self.pinned_buffers_peak = max(self.pinned_buffers_peak, 1)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready)
            pinned.copy_(output, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
        copied.synchronize()
        return pinned

    def report(self):
        return dict(local_blocks=list(self.local_blocks), collective_blocks=list(self.collective_blocks),
                    worker_issues_collectives=False, worker_count=1,
                    pinned_buffers_peak=self.pinned_buffers_peak,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.rank),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.rank),
                    temporal_cache_changed=False, spatial_padding_changed=False,
                    output_warmup_frames_removed=9, thread_local_bf16_autocast=True)

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.decode_stream.synchronize()
            self.copy_stream.synchronize()
            self.closed = True


class GeneratorOverlapAdapter:
    """Patch only output scheduling in the frozen generator, with AST guards."""
    def __init__(self, generator, create_pipeline):
        self.generator = generator
        self.original = generator.generate
        self.create = create_pipeline
        self.installed = False

    def install(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.original)))
        function = tree.body[0]
        stamp = lambda node: ast.dump(node, include_attributes=False)
        sequence = ast.parse('''
if f == 0:
    _latent = latent
    with diagnostics.phase("vae_decode"):
        _videos = vae.decode(_latent.squeeze(0))
else:
    _latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
    with diagnostics.phase("vae_decode"):
        _videos = vae.decode(_latent.squeeze(0))[:, :, 9:]
pre_latent = latent
repro.tensor('decoded_block', _videos, block=_)
with diagnostics.phase("video_d2h"):
    gen_video_list.append(_videos.cpu())
_p7_trace("block_done", sync=True, block=_)
if args.dura_print:
    torch.cuda.synchronize()
    if rank == 0:
        t2 = time.time()
        dura = blksz_lst[f] * vae_stride[0] / fps * 1000
        print(f"Done Block {_}: duration {dura}ms video cost {(t2 - t1) * 1000:.2f} ms")
''').body
        loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == '_'
                 and stamp(n.iter) == stamp(ast.parse('range(iter_total_num)', mode='eval').body)]
        if len(loops) != 1:
            raise ValueError('One original generation block loop required')
        loop = loops[0]
        matches = []
        for container in ast.walk(loop):
            body = getattr(container, 'body', None)
            if isinstance(body, list):
                for index in range(len(body)-len(sequence)+1):
                    if [stamp(n) for n in body[index:index+len(sequence)]] == [stamp(n) for n in sequence]:
                        matches.append((body, index))
        if len(matches) != 1:
            raise ValueError('Original VAE/output/sync sequence required')
        body, index = matches[0]
        body[index:index+len(sequence)] = ast.parse('''
if f == 0:
    _latent = latent
else:
    _latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
pre_latent = latent
_r2_pipeline.consume(_, _latent, gen_video_list)
''').body
        loop.body.insert(0, ast.parse('_r2_pipeline.begin_block(_)').body[0])
        counts = dict(init=0, flush=0)
        class Insert(ast.NodeTransformer):
            def visit_Assign(self, node):
                if stamp(node) == stamp(ast.parse('gen_video_list = []').body[0]):
                    counts['init'] += 1
                    return [node, ast.parse('_r2_pipeline = _r2_create_pipeline(vae, rank, target_frames, iter_total_num)').body[0]]
                return self.generic_visit(node)
            def visit_For(self, node):
                node = self.generic_visit(node)
                if node is loop:
                    counts['flush'] += 1
                    return [node, ast.parse('_r2_pipeline.flush(gen_video_list)').body[0]]
                return node
        tree = ast.fix_missing_locations(Insert().visit(tree))
        self.transformed_tree = tree
        if counts != dict(init=1, flush=1):
            raise ValueError('Exactly one pipeline init and EOS flush required')
        # Use the live module globals so the existing rotary task-reset trace
        # and bounded-audio wrapper still take effect after installation.
        namespace = self.generator.__dict__
        self.had_helper = '_r2_create_pipeline' in namespace
        self.previous_helper = namespace.get('_r2_create_pipeline')
        namespace['_r2_create_pipeline'] = self.create
        exec(compile(tree, inspect.getsourcefile(self.original), 'exec'), namespace)
        self.generator.generate = wraps(self.original)(namespace[function.name])
        self.installed = True

    def close(self):
        if self.installed:
            self.generator.generate = self.original
            if self.had_helper:
                self.generator.__dict__['_r2_create_pipeline'] = self.previous_helper
            else:
                self.generator.__dict__.pop('_r2_create_pipeline', None)
        self.installed = False
