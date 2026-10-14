import os
import numpy as np
import random
import math
import time
from tqdm import tqdm
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms
import torchaudio
from video_timing import target_frame_count, required_blocks
from scripts.p10_inference_profile import InferenceProfiler
from scripts.p10_repro_trace import ReproTrace
from src.audio_analysis.bounded_audio import BoundedAudioConditioner
from src.audio_analysis.streaming_audio import ReplayPreparedPCM, StreamingAudioConditioner
from control_spec import (active_edit, conditioning_prompt, effective_schedule,
                          normalize_case, normalize_edits)
import torchaudio.transforms as T

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import *

from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from diffusers.utils import export_to_video

from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from fp4_gemm import FP4GemmOptions, enable_fp4_gemm


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _p7_trace(stage, *, sync=False, **details):
    """Write opt-in P7 timing events; a GPU sync makes stage boundaries measurable."""
    destination = os.environ.get("LIVEACT_P7_TIMELINE")
    if not destination or int(os.environ.get("RANK", "0")) != 0:
        return
    if sync:
        torch.cuda.synchronize()
    event = {"stage": stage, "wall_ns": time.time_ns(),
             "monotonic_ns": time.perf_counter_ns(), "cuda_synchronized": sync,
             **details}
    with Path(destination).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt, image and audio"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--offload_cache",
        action="store_true",
        default=False,
        help="Whether to place kv cache on CPU.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        default=False,
        help="Whether to store kv cache in FP8 and dequantize to BF16 on use.")
    parser.add_argument(
        "--fp8_gemm",
        action="store_true",
        default=False,
        help="Whether to enable FP8 GEMM for the model.")
    parser.add_argument(
        "--fp4_gemm",
        action="store_true",
        default=False,
        help="Whether to enable FP4 GEMM for the model.")
    parser.add_argument(
        "--block_offload",
        action="store_true",
        default=False,
        help="Whether to offload WanModel blocks to CPU between block forwards.")
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="The target fps.")
    parser.add_argument(
        "--audio_cfg",
        type=float,
        default=1.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--dura_print",
        action="store_true",
        default=False,
        help="Whether print duration for every block.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--steam_audio",
        action="store_true",
        default=False,
        help="Whether inference with steaming audio.")
    parser.add_argument(
        "--mean_memory",
        action="store_true",
        default=False,
        help="Whether inference with mean memory strategy.")
    parser.add_argument("--audio_lookahead", type=float, default=None,
                        help="Offline bounded-encoder quality test: extra future seconds beyond the conditioning window and its two right-context features")
    parser.add_argument("--audio_history", type=float, default=4.0,
                        help="History seconds before the condition end for bounded-encoder tests")
    parser.add_argument("--audio_stream_pcm", action="store_true",
                        help="Experimental 200 ms PCM16 replay with stateful tempo; requires audio_lookahead, not live-service acceptance")
    parser.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep baseline cuDNN autotuning by default; disabling is a reproducibility diagnostic, not guaranteed deterministic execution')
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")

    args = parser.parse_args()
    if args.audio_stream_pcm and args.audio_lookahead is None:
        parser.error("audio_stream_pcm requires audio_lookahead")
    if args.audio_lookahead is not None:
        if args.steam_audio:
            parser.error("audio_lookahead and steam_audio are mutually exclusive")
        if (not math.isfinite(args.audio_lookahead) or args.audio_lookahead < 0 or
                not math.isfinite(args.audio_history) or args.audio_history <= 0):
            parser.error("Invalid bounded-audio history or lookahead")

    return args


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _p7_trace("generate_enter")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    if world_size > 1:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=world_size,
        )

    if world_size > 1:
        from model_liveact.model_memory_sp import WanModel
    else:
        from model_liveact.model_memory import WanModel

    width, height = [int(_) for _ in args.size.split('*')]
    fps = args.fps
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    timesteps = [torch.tensor([_]).to(device, dtype=torch.float32) for _ in [1000.0, 937.5, 833.33333333, 0.0]]
    blksz_lst = [6, 8]
    frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
    kv_cache_tokens = frame_len * sum(blksz_lst) // world_size
    kv_cache_device = 'cpu' if args.offload_cache else device
    kv_cache_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16
    kv_scale_shape = (1, kv_cache_tokens, 40, 1)
    kv_cache = \
        {
            i: {
                layer_id: {
                    'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'mean_memory': args.mean_memory,
                    'offload_cache': args.offload_cache,
                    'fp8_kv_cache': args.fp8_kv_cache,
                }
                for layer_id in range(40)
            } for i in range(len(timesteps) - 1)
        }
    if args.audio_cfg > 1.0:
        kv_cache_null_audio = \
            {
                i: {layer_id: {
                    'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'mean_memory': args.mean_memory,
                    'offload_cache': args.offload_cache,
                    'fp8_kv_cache': args.fp8_kv_cache,
                } for layer_id in range(40)} for i in range(len(timesteps) - 1)
            }

    _p7_trace("model_load_start")
    wan_i2v_model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    wan_i2v_model = wan_i2v_model.to(dtype=torch.bfloat16)
    for n in range(40):
        wan_i2v_model.blocks[n].self_attn.init_kvidx(frame_len, world_size)

    vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=device,
                   use_lightvae=False, parallel=(world_size > 1))

    clip = CLIPModel(
        checkpoint_path=os.path.join(args.ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
        tokenizer_path=os.path.join(args.ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=device)
    clip.model = clip.model.to(device, dtype=torch.bfloat16)

    text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cpu' if args.t5_cpu else device,
                                  checkpoint_path=os.path.join(args.ckpt_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
                                  tokenizer_path=os.path.join(args.ckpt_dir, 'google/umt5-xxl'))

    audio_encoder = Wav2Vec2Model.from_pretrained(
        args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device, dtype=torch.bfloat16).eval()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)

    audio_encoder.feature_extractor._freeze_parameters()
    wan_i2v_model.freqs = wan_i2v_model.freqs.to(device)
    for _model in [wan_i2v_model, clip.model, audio_encoder, vae.model]:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    if args.fp8_gemm:
        enable_fp8_gemm(wan_i2v_model, options=FP8GemmOptions())
    if args.fp4_gemm:
        print("Enabling FP4 GEMM acceleration...")
        def filter_fn(name, module):
            if "blocks." not in name:
                return False
            return True
        enable_fp4_gemm(wan_i2v_model, options=FP4GemmOptions(), module_filter=filter_fn)
    if args.block_offload:
        for name, child in wan_i2v_model.named_children():
            if name != 'blocks':
                child.to(device)
        wan_i2v_model.enable_block_offload(
            onload_device=torch.device(f"cuda:{device}"),
        )
    else:
        wan_i2v_model = wan_i2v_model.to(device)
    wan_i2v_model.eval()
    wan_i2v_model = torch.compile(wan_i2v_model)

    vae.model.eval()
    vae.encode = torch.compile(vae.encode)

    torch_gc()
    _p7_trace("model_ready", sync=True)
    diagnostics = InferenceProfiler(wan_i2v_model)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    for case_index, data in enumerate(input_data):
        repro = ReproTrace.from_environment(case_index=case_index, rank=rank, device=device,
            settings={'cudnn_benchmark': torch.backends.cudnn.benchmark,
                      'cudnn_deterministic': torch.backends.cudnn.deterministic,
                      'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                      'seed': args.seed, 'rank': rank})
        _p7_trace("condition_start")
        control = normalize_case(data)
        image_path = control['reference_image']
        audio_path = control['audio']
        out_path = os.path.basename(image_path).split('.')[0] + '_' + os.path.basename(audio_path).split('.')[0] + '.mp4'
        prompt = control['main_prompt']
        persistent_prompt = control['persistent_prompt']
        edits = control['edits']

        main_conditioning = conditioning_prompt(prompt, persistent_prompt=persistent_prompt)
        context = [text_encoder(texts=main_conditioning, device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)]
        edit_prompts = {text: text_encoder(texts=conditioning_prompt(prompt, text, persistent_prompt),
                                          device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)
                        for text in dict.fromkeys(edit['prompt'] for edit in edits)}
        _p7_trace("t5_done", sync=True)
        repro.tensor('text_context', context[0])

        image = Image.open(image_path).convert("RGB")
        cond_image = transform(image).unsqueeze(1).unsqueeze(0).to(device, torch.bfloat16)  # 1 C 1 H W
        clip.model.to(device)
        clip_context = clip.visual(cond_image)  # 1, 257, 1280
        clip.model.cpu()
        torch_gc()
        _p7_trace("clip_done", sync=True)
        repro.tensor('clip_context', clip_context)

        audio_ori, sr_ori = torchaudio.load(audio_path)  # y: [channels, time]
        def resample_audio(audio, sr, fps):
            rate = 25 / fps
            effects = [["tempo", f"{rate}"], ]
            y, sr = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
            resampler = T.Resample(sr, 16000)
            return resampler(y) * 3.0, 16000

        bounded_audio = None
        streaming_pcm = None
        if args.audio_stream_pcm:
            if sr_ori != 16000 or audio_ori.shape[0] != 1:
                raise ValueError("PCM replay requires mono 16 kHz input, matching the live wire format")
            integer_pcm = torch.round(audio_ori[0] * 32768).clamp(-32768, 32767).to(torch.int16)
            if not torch.equal(integer_pcm.float() / 32768, audio_ori[0]):
                raise ValueError("Replay input must already be PCM16; silent quantization is not allowed")
            streaming_pcm = ReplayPreparedPCM(
                integer_pcm.numpy().astype('<i2').tobytes(), fps=fps,
                sox=Path(__file__).resolve().parent / '.liveact-env/bin/sox')
            bounded_audio = StreamingAudioConditioner(
                streaming_pcm, lambda pcm: get_embedding(
                    pcm, wav2vec_feature_extractor, audio_encoder, device=device),
                history_seconds=args.audio_history,
                extra_lookahead_seconds=args.audio_lookahead, device=device,
                log_path=Path.cwd() / f"streaming-audio-{case_index}-rank-{rank}.jsonl",
                repro_trace=repro)
        else:
            audio, sr = resample_audio(audio_ori, sr_ori, fps)
            if args.audio_lookahead is None:
                audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
            else:
                bounded_audio = BoundedAudioConditioner(
                    audio[0], lambda pcm: get_embedding(
                        pcm, wav2vec_feature_extractor, audio_encoder, device=device),
                    fps=fps, history_seconds=args.audio_history,
                    extra_lookahead_seconds=args.audio_lookahead, device=device,
                    log_path=Path.cwd() / f"bounded-audio-{case_index}-rank-{rank}.jsonl",
                    repro_trace=repro)
        audio_len = audio_ori.size(1) / sr_ori
        _p7_trace("wav2vec_done", sync=True,
                  mode=("packet_replay_deferred_to_blocks" if streaming_pcm else
                        "bounded_encoding_deferred_to_blocks" if bounded_audio else "full_audio_encoded"))

        ref_target_masks = torch.ones(3, height // vae_stride[1], width // vae_stride[2]).to(device, torch.bfloat16)
        frame_num = (sum(blksz_lst) - 1) * 4 + 1
        msk = get_msk(frame_num, cond_image, vae_stride, device)

        def get_y(frame_num):
            video_frames = torch.zeros(
                1, cond_image.shape[1], frame_num - cond_image.shape[2], height, width
            ).to(cond_image.device, cond_image.dtype)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            y = vae.encode(padding_frames_pixels_values.to(vae.device)).to(wan_i2v_model.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)
            return y

        y = get_y(frame_num)
        _p7_trace("vae_condition_done", sync=True)
        repro.tensor('vae_condition', y)

        target_frames = target_frame_count(audio_len, fps)
        iter_total_num = required_blocks(target_frames, blksz_lst, vae_stride[0])
        edits = normalize_edits(edits, block_count=iter_total_num)
        if rank == 0:
            (Path.cwd() / f"control-{case_index}.json").write_text(json.dumps({
                **control, "edits": edits, "fps": fps, "size": args.size,
                "seed": args.seed, "target_frames": target_frames,
                "block_count": iter_total_num,
                "effective_controls": effective_schedule(edits, block_count=iter_total_num,
                                                          main_prompt=prompt,
                                                          persistent_prompt=persistent_prompt),
            }, ensure_ascii=False, indent=2) + "\n")
        print('----iter_total_num=', iter_total_num)
        gen_video_list = []
        torch.manual_seed(args.seed)
        repro.rng('seeded_rng')
        for _ in range(iter_total_num):
            _p7_trace("block_start", block=_)
            t1 = time.time()
            diagnostics.begin_block(_)
            audio_start_idx, audio_end_idx = 0, frame_num
            if (_ - 1) * blksz_lst[-1] * vae_stride[0] > 0:
                audio_start_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]
                audio_end_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]

            repro.rng('before_audio_rng', block=_)
            if bounded_audio is not None:
                audio_embs = bounded_audio(audio_start_idx, audio_end_idx, block=_)
            elif not args.steam_audio:
                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device)
            else:
                audio, sr = resample_audio(
                    audio_ori[:1, int(sr_ori*(audio_start_idx/fps)):int(sr_ori*((audio_end_idx+2)/fps))], sr_ori, fps
                )
                audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
                audio_embs = get_audio_emb(audio_embedding, 0, frame_num, device)

            repro.tensor('audio_condition', audio_embs, block=_)
            repro.rng('after_audio_rng', block=_)

            y_cut = y[:, :, :frame_num // 4 + 1, ...]

            selected_edit = active_edit(edits, _)
            _context = ([edit_prompts[selected_edit['prompt']]]
                        if selected_edit else context)

            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                f = _ if _ <= 1 else 1
                latent = torch.randn(16, blksz_lst[f], height // vae_stride[1], width // vae_stride[2],
                                     dtype=torch.bfloat16, device=device)
                repro.tensor('initial_latent', latent, block=_)
                for i in tqdm(range(len(timesteps) - 1)):
                    timestep = timesteps[i]
                    arg_c = {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': audio_embs, 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                    with diagnostics.phase("dit_step", step=i):
                        noise_pred = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache[i],
                                                   skip_audio=False if i in [1, 2] else True, **arg_c)[0]

                    if args.audio_cfg>1.0 and i in [1, 2]:
                        arg_null_audio = \
                            {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': torch.zeros_like(audio_embs), 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                        noise_pred_drop_audio = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache_null_audio[i],
                                                              **arg_null_audio)[0]
                        noise_pred = noise_pred_drop_audio + args.audio_cfg * (noise_pred - noise_pred_drop_audio)

                    dt = timesteps[i] - timesteps[i + 1]
                    repro.tensor('noise_prediction', noise_pred, block=_, step=i)
                    dt = dt / 1000
                    # latent = latent + (-noise_pred) * dt[0]
                    x0_pred = latent + (-noise_pred) * (timesteps[i][0]/1000 - 0.0)
                    latent = (1-timesteps[i+1][0]/1000)*x0_pred + torch.randn_like(x0_pred)*(timesteps[i+1][0]/1000)
                    repro.tensor('updated_latent', latent, block=_, step=i)
                    repro.rng('after_step_rng', block=_, step=i)

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
                diagnostics.end_block()
        if streaming_pcm is not None:
            streaming_pcm.close()
        torch.cuda.synchronize()
        # torch_gc()

        if rank == 0:
            videos = (torch.concat(gen_video_list, dim=2).permute((0, 2, 3, 4, 1))[0] + 1.0) / 2
            if videos.shape[0] < target_frames:
                raise RuntimeError(f"Generated {videos.shape[0]} frames; need {target_frames}")
            videos = videos[:target_frames]
            video_path = 'tmp.mp4'
            _p7_trace("video_encode_start")
            export_to_video(videos[:, ...].float().cpu().numpy(), video_path, fps=fps)
            _p7_trace("video_encode_done")
            add_audio_to_video(video_path, audio_path, out_path)
            _p7_trace("audio_mux_done")

        torch.cuda.synchronize()
        # torch_gc()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
