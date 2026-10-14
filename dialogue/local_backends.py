"""CPU-only, local-path backends; no remote inference or implicit model fetches.

These adapters do not themselves connect TTS to LiveAct or claim realtime.
Load models in a Slurm allocation, never on a login node.
"""
from pathlib import Path
import os


def local_file(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Required local model file missing: {path}')
    return path


def offline_environment():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def sentences(tokens, *, max_chars=48):
    """Yield speakable prefixes before the full LLM response is available."""
    if type(max_chars) is not int or max_chars < 8:
        raise ValueError('A speech phrase needs a valid character budget')
    pending = ''
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError('Text deltas must be strings')
        for char in token:
            pending += char
            if char in '。！？!?；;\n' or len(pending) >= max_chars:
                if pending.strip():
                    yield pending.strip()
                pending = ''
    if pending.strip():
        yield pending.strip()


class LocalASR:
    def __init__(self, directory, *, threads=8):
        offline_environment()
        directory = Path(directory).resolve()
        for filename in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
            local_file(directory / filename)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(directory), device='cpu', compute_type='int8',
                                  cpu_threads=threads, num_workers=1, local_files_only=True)

    def transcribe(self, pcm16):
        if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
            raise ValueError('ASR requires little-endian mono PCM16 at 16 kHz')
        import numpy as np
        audio = np.frombuffer(pcm16, dtype='<i2').astype(np.float32) / 32768
        segments, info = self.model.transcribe(audio, language='zh', beam_size=1,
                                              condition_on_previous_text=False,
                                              temperature=0, vad_filter=True)
        rows = [dict(start=part.start, end=part.end, text=part.text,
                     no_speech_probability=part.no_speech_prob) for part in segments]
        return dict(text=''.join(part['text'] for part in rows).strip(), segments=rows,
                    language=info.language, input_seconds=len(audio)/16000)


class LocalLLM:
    def __init__(self, model, *, threads=8, context=2048):
        offline_environment()
        path = local_file(model)
        from llama_cpp import Llama
        self.model = Llama(model_path=str(path), n_gpu_layers=0, n_ctx=context,
                           n_threads=threads, n_threads_batch=threads, n_batch=128,
                           verbose=False)
        self.context = context

    def prepare_context(self, messages, *, max_tokens=128):
        messages = [dict(row) for row in messages]
        # Budget actual content tokens plus conservative ChatML framing. Keep
        # the system and latest user request, dropping complete old pairs.
        def budget():
            return sum(len(self.model.tokenize(row['content'].encode('utf-8'), add_bos=False))
                       + 64 for row in messages) + 64 + max_tokens
        while budget() > self.context and len(messages) > 2:
            del messages[1:3]
        if budget() > self.context:
            raise ValueError('Latest user input exceeds the local LLM context budget')
        return messages

    def stream(self, messages, *, max_tokens=128):
        response = self.model.create_chat_completion(messages=messages, stream=True,
                                                     max_tokens=max_tokens,
                                                     temperature=0.4, top_p=0.9)
        try:
            for event in response:
                text = event['choices'][0].get('delta', {}).get('content')
                if text:
                    yield text
        finally:
            close = getattr(response, 'close', None)
            if close:
                close()


class LocalTTS:
    sample_rate = 16000

    def __init__(self, directory, *, threads=4, voice_id=None):
        offline_environment()
        directory = Path(directory).resolve()
        config = local_file(directory / 'config.json')
        weight = local_file(directory / 'kokoro-v1_1-zh.pth')
        from dialogue.voices import verify_voice
        self.directory = directory
        self.voice_info = verify_voice(directory, voice_id)
        self.voice_id = self.voice_info['id']
        import torch
        torch.set_num_threads(threads)
        from kokoro import KModel, KPipeline
        model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh', config=str(config),
                       model=str(weight)).to('cpu').eval()
        self.pipeline = KPipeline(lang_code='z', repo_id=str(directory), model=model,
                                  device='cpu')
        self._voices = {}
        self.voice = self._load_voice(self.voice_info)

    def _load_voice(self, info):
        import torch
        if info['id'] not in self._voices:
            voice = torch.load(info['path'], map_location='cpu', weights_only=True)
            if not isinstance(voice, torch.Tensor) or voice.ndim != 3 or voice.shape[1:] != (1, 256):
                raise ValueError('Unexpected Kokoro preset tensor dimensions')
            if not torch.isfinite(voice).all():
                raise ValueError('Kokoro preset voice contains non-finite values')
            self._voices[info['id']] = voice
        return self._voices[info['id']]

    def stream(self, text, *, voice_id=None):
        import torch
        import torchaudio
        import numpy as np
        # A per-utterance selection does not change another utterance's voice.
        from dialogue.voices import verify_voice
        voice = self.voice if voice_id is None else self._load_voice(verify_voice(self.directory, voice_id))
        for result in self.pipeline(text, voice=voice, speed=1):
            if result.audio is None:
                raise ValueError('TTS produced no waveform')
            waveform = result.audio.detach().to(device='cpu', dtype=torch.float32).flatten()
            if not torch.isfinite(waveform).all() or waveform.numel() == 0:
                raise ValueError('TTS produced an invalid waveform')
            # Kokoro emits 24 kHz. Each yielded sentence is an independent
            # synthesized waveform, resampled once, without changing playback
            # speed. LiveAct's internal tempo preparation is a later stage.
            waveform = torchaudio.functional.resample(waveform, 24000, 16000)
            pcm = np.clip(np.rint(waveform.numpy()*32768), -32768, 32767).astype('<i2')
            yield pcm.tobytes()
