"""Bounded PCM session protocol for the P10 experimental input path.

Wire audio packets are little-endian uint32 sequence and start-sample fields
followed by mono signed 16-bit PCM at 16 kHz. Acknowledgements are sent only
after the packet is committed to the private WAV file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import struct
import time
import wave
from control_spec import block_for_frame
from video_timing import required_blocks, target_frame_count


HEADER = struct.Struct("<II")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 3200
RESUME_SECONDS = 30
TERMINAL = {"finished", "failed"}


class ProtocolError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class AudioSession:
    def __init__(self, root, *, max_seconds=15, clock=time.monotonic):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.clock = clock
        self.max_samples = round(max_seconds * SAMPLE_RATE)
        self.token = secrets.token_urlsafe(32)
        self.state = "created"
        self.next_seq = 0
        self.received_samples = 0
        self.control_version = 0
        self.controls = []
        self.started_at = None
        self.resume_deadline = None
        self.last_packet = None
        self.writer = wave.open(str(self.root / "input.wav.part"), "wb")
        self.writer.setnchannels(1)
        self.writer.setsampwidth(2)
        self.writer.setframerate(SAMPLE_RATE)
        self._log("created")

    def _log(self, event, **fields):
        record = dict(event=event, state=self.state, time=time.time(),
                      next_seq=self.next_seq, received_samples=self.received_samples,
                      control_version=self.control_version, **fields)
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def snapshot(self):
        return {"state": self.state, "next_seq": self.next_seq,
                "received_samples": self.received_samples,
                "duration_seconds": self.received_samples / SAMPLE_RATE,
                "control_version": self.control_version,
                "buffer_depth": 0, "max_seconds": self.max_samples / SAMPLE_RATE}

    def connect(self, token):
        if not secrets.compare_digest(token, self.token):
            raise ProtocolError("Invalid session token", "unauthorized")
        if self.state == "created":
            self.state = "priming"
        elif self.state == "interrupted" and self.resume_deadline is not None:
            if self.clock() > self.resume_deadline:
                self.fail("Reconnect timeout")
                raise ProtocolError("Reconnect window expired", "expired")
            self.state = "streaming" if self.received_samples >= SAMPLE_RATE // 2 else "priming"
            self.resume_deadline = None
            self.started_at = self.clock() - self.received_samples / SAMPLE_RATE
        else:
            raise ProtocolError("Session is not connectable", "invalid_state")
        self._log("connected")
        return self.snapshot()

    def accept_audio(self, packet):
        if self.state not in {"priming", "streaming"}:
            raise ProtocolError("Session is not receiving audio", "invalid_state")
        if len(packet) < HEADER.size + 2 or (len(packet) - HEADER.size) % 2:
            raise ProtocolError("Malformed PCM packet", "malformed")
        seq, start_sample = HEADER.unpack_from(packet)
        pcm = packet[HEADER.size:]
        samples = len(pcm) // 2
        if samples > CHUNK_SAMPLES:
            raise ProtocolError("PCM chunk exceeds 200 ms", "chunk_too_large")
        digest = hashlib.sha256(packet).digest()
        if seq == self.next_seq - 1 and self.last_packet == digest:
            return dict(self.snapshot(), duplicate=True)
        if seq != self.next_seq:
            raise ProtocolError("Missing, stale or reordered PCM sequence", "sequence")
        if start_sample != self.received_samples:
            raise ProtocolError("PCM timestamp is not contiguous", "timestamp")
        if self.received_samples + samples > self.max_samples:
            raise ProtocolError("Audio duration exceeds session limit", "duration")
        if self.started_at is None:
            self.started_at = self.clock()
        if self.started_at is not None and (
                self.received_samples + samples) / SAMPLE_RATE > (
                self.clock() - self.started_at + 0.5):
            raise ProtocolError("Audio is arriving faster than the bounded window", "backpressure")
        self.writer.writeframesraw(pcm)
        self.received_samples += samples
        self.next_seq += 1
        self.last_packet = digest
        if self.received_samples >= SAMPLE_RATE // 2:
            self.state = "streaming"
        self._log("audio_committed", seq=seq, start_sample=start_sample,
                  samples=samples)
        return self.snapshot()

    def add_control(self, event):
        if self.state not in {"priming", "streaming"}:
            raise ProtocolError("Session is not receiving controls", "invalid_state")
        if not isinstance(event, dict):
            raise ProtocolError("Control event must be an object", "control")
        prompt = event.get("prompt")
        kind = event.get("kind")
        if kind not in {"action", "emotion"} or not isinstance(prompt, str):
            raise ProtocolError("Invalid control kind or prompt", "control")
        prompt = prompt.strip()
        if not prompt or len(prompt) > 1000 or len(self.controls) >= 8:
            raise ProtocolError("Invalid control prompt or limit", "control")
        min_block = block_for_frame(self.received_samples * 20 // SAMPLE_RATE) + 1
        max_block = required_blocks(
            target_frame_count(self.max_samples / SAMPLE_RATE, 20)) - 1
        block = event.get("effective_from_block", min_block)
        if type(block) is not int or block < min_block or block > max_block:
            raise ProtocolError("Control must target a future block", "control_time")
        self.control_version += 1
        item = {"version": self.control_version, "kind": kind,
                "prompt": prompt, "effective_from_block": block}
        self.controls.append(item)
        self._log("control_accepted", version=self.control_version,
                  effective_from_block=block, kind=kind)
        return dict(self.snapshot(), accepted_control=item)

    def finish(self):
        if self.state not in {"priming", "streaming"}:
            raise ProtocolError("Session cannot finish", "invalid_state")
        if self.received_samples < SAMPLE_RATE // 2:
            raise ProtocolError("Audio must be at least 0.5 seconds", "duration")
        self.writer.close()
        self.writer = None
        (self.root / "input.wav.part").replace(self.root / "input.wav")
        self.state = "draining"
        self._log("input_finished", duration_seconds=self.received_samples / SAMPLE_RATE)
        return self.snapshot()

    def disconnect(self):
        if self.state in {"priming", "streaming"}:
            self.state = "interrupted"
            self.resume_deadline = self.clock() + RESUME_SECONDS
            self._log("disconnected", resume_deadline_monotonic=self.resume_deadline)

    def fail(self, reason):
        if self.state in TERMINAL:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.state = "failed"
        self._log("failed", reason=reason)

    def cancel(self):
        if self.state in TERMINAL:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.state = "interrupted"
        self.resume_deadline = None
        self._log("cancelled")

    def finish_output(self):
        if self.state != "draining":
            raise ProtocolError("Output is not draining", "invalid_state")
        self.state = "finished"
        self._log("finished")
