"""Shared, CUDA-free P8 text-control semantics for offline and web inference."""

import ast
import math

from video_timing import required_blocks, target_frame_count


MAX_EDITS = 8
MAX_PROMPT_CHARS = 2000
EDIT_TYPES = frozenset({"general", "action", "emotion"})
FIRST_BLOCK_FRAMES = 21
LATER_BLOCK_FRAMES = 32


def block_for_frame(frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("Frame index must be a non-negative integer")
    return 0 if frame < FIRST_BLOCK_FRAMES else 1 + (frame - FIRST_BLOCK_FRAMES) // LATER_BLOCK_FRAMES


def block_frame_bounds(block, total_frames):
    if type(block) is not int or block < 0 or type(total_frames) is not int or total_frames <= 0:
        raise ValueError("Invalid block or frame count")
    start = 0 if block == 0 else FIRST_BLOCK_FRAMES + (block - 1) * LATER_BLOCK_FRAMES
    end = min(total_frames, FIRST_BLOCK_FRAMES + block * LATER_BLOCK_FRAMES)
    if start >= total_frames:
        raise ValueError("Block is outside the generated video")
    return start, end


def seconds_to_blocks(start_seconds, end_seconds, *, fps, audio_seconds):
    """Map a requested half-open time span to all video blocks it touches.

    Because one prompt conditions an entire block, the effective start may be
    earlier and the effective end later than the requested second values.
    """
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (start_seconds, end_seconds, fps, audio_seconds)):
        raise ValueError("Time values must be finite numbers")
    if not 0 <= start_seconds < end_seconds <= audio_seconds + 0.001:
        raise ValueError("Edit time must lie inside the audio duration")
    frames = target_frame_count(audio_seconds, fps)
    start_frame = min(frames - 1, math.floor(start_seconds * fps + 1e-9))
    end_frame = min(frames, math.ceil(end_seconds * fps - 1e-9)) - 1
    if end_frame < start_frame:
        end_frame = start_frame
    return block_for_frame(start_frame), block_for_frame(end_frame)


def _legacy_edit_items(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("Legacy edit_prompt must be an object")
    edits = []
    for key, prompt in mapping.items():
        try:
            pair = ast.literal_eval(key)
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError("Invalid legacy edit interval") from error
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Invalid legacy edit interval")
        edits.append({"start_block": pair[0], "end_block": pair[1],
                      "prompt": prompt, "type": "general"})
    return edits


def normalize_edits(raw, *, block_count=None, fps=None, audio_seconds=None):
    """Return ordered edits; a later edit wins when closed block spans overlap."""
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > MAX_EDITS:
        raise ValueError(f"Edits must be a list of at most {MAX_EDITS} items")
    if block_count is not None and (type(block_count) is not int or block_count <= 0):
        raise ValueError("Invalid block count")
    normalized = []
    for item in raw:
        if isinstance(item, list) and len(item) == 3:
            item = {"start_block": item[0], "end_block": item[1],
                    "prompt": item[2], "type": "general"}
        if not isinstance(item, dict):
            raise ValueError("Each edit must be an object or legacy three-item list")
        uses_seconds = "start_seconds" in item or "end_seconds" in item
        uses_blocks = "start_block" in item or "end_block" in item
        if uses_seconds == uses_blocks:
            raise ValueError("Use either seconds or block indices for each edit")
        if uses_seconds:
            if fps is None or audio_seconds is None:
                raise ValueError("Audio duration and FPS are needed for second-based edits")
            start, end = seconds_to_blocks(item.get("start_seconds"), item.get("end_seconds"),
                                           fps=fps, audio_seconds=audio_seconds)
        else:
            start, end = item.get("start_block"), item.get("end_block")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("Each edit needs a valid closed block interval")
        if block_count is not None and end >= block_count:
            raise ValueError(f"Edit block must be smaller than {block_count}")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
            raise ValueError("Each edit needs a non-empty prompt of at most 2000 characters")
        kind = item.get("type", "general")
        if not isinstance(kind, str) or kind not in EDIT_TYPES:
            raise ValueError("Edit type must be general, action, or emotion")
        normalized.append({"start_block": start, "end_block": end,
                           "prompt": prompt.strip(), "type": kind})
    return normalized


def normalize_case(data, *, block_count=None):
    """Accept canonical P8 inputs while retaining upstream JSON compatibility."""
    if not isinstance(data, dict):
        raise ValueError("Control case must be an object")
    image = data.get("reference_image", data.get("cond_image"))
    audio = data.get("audio", data.get("cond_audio"))
    prompt = data.get("main_prompt", data.get("prompt"))
    if not isinstance(image, str) or not image or not isinstance(audio, str) or not audio:
        raise ValueError("A reference image and audio path are required")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= MAX_PROMPT_CHARS:
        raise ValueError("A non-empty main prompt of at most 2000 characters is required")
    persistent_prompt = normalize_persistent_prompt(data.get("persistent_prompt"))
    if "edits" in data:
        raw = data["edits"]
    else:
        raw = _legacy_edit_items(data.get("edit_prompt", {}))
    edits = normalize_edits(raw, block_count=block_count)
    conditioning_prompt(prompt.strip(), persistent_prompt=persistent_prompt)
    for edit in edits:
        conditioning_prompt(prompt.strip(), edit["prompt"], persistent_prompt)
    return {"reference_image": image, "audio": audio, "main_prompt": prompt.strip(),
            "persistent_prompt": persistent_prompt, "edits": edits}


def normalize_persistent_prompt(value):
    """An optional condition that remains active while local edits change."""
    if value is None:
        return ""
    if not isinstance(value, str) or len(value.strip()) > MAX_PROMPT_CHARS:
        raise ValueError("Persistent prompt must be text of at most 2000 characters")
    return value.strip()


def conditioning_prompt(main_prompt, edit_prompt=None, persistent_prompt=""):
    """Preserve legacy replacement unless a persistent condition is supplied."""
    if not persistent_prompt:
        return edit_prompt if edit_prompt is not None else main_prompt
    parts = [main_prompt, persistent_prompt]
    if edit_prompt is not None:
        parts.append(edit_prompt)
    combined = "，".join(part.strip().rstrip("，。；; ") for part in parts)
    if len(combined) > MAX_PROMPT_CHARS:
        raise ValueError("Combined prompt must be at most 2000 characters")
    return combined


def active_edit(edits, block):
    """Closed intervals and last-in-list priority are identical in both runtimes."""
    if type(block) is not int or block < 0:
        raise ValueError("Block index must be a non-negative integer")
    for edit in reversed(edits):
        if edit["start_block"] <= block <= edit["end_block"]:
            return edit
    return None


def effective_schedule(edits, *, block_count, main_prompt, persistent_prompt=""):
    if type(block_count) is not int or block_count <= 0:
        raise ValueError("Invalid block count")
    return [{"block": block,
             "prompt": conditioning_prompt(main_prompt,
                                           selected["prompt"] if selected else None,
                                           persistent_prompt),
             "type": (selected["type"] if selected else "main")}
            for block in range(block_count)
            for selected in [active_edit(edits, block)]]


def block_count_for_audio(audio_seconds, fps):
    return required_blocks(target_frame_count(audio_seconds, fps))
