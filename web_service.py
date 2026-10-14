"""HTTP validation and task state, independent of CUDA and model imports."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import threading
import time
import uuid
import warnings
import wave

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, send_from_directory, session
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException
from control_spec import (conditioning_prompt, effective_schedule,
                          normalize_edits, normalize_persistent_prompt)
from video_timing import required_blocks, target_frame_count

ROOT = Path(__file__).resolve().parent
TASK_ID = re.compile(r"[a-f0-9]{32}")
TERMINAL = {"finished", "failed", "interrupted"}

class AdmissionError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code

class ServiceState:
    def __init__(self, root, *, size="416*720", max_seconds=15, max_upload_mib=32, allowed_fps=(20,)):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.tasks_root = self.root / "tasks"
        self.tasks_root.mkdir(mode=0o700)
        self.size = size
        self.max_seconds = max_seconds
        self.max_upload_mib = max_upload_mib
        self.allowed_fps = tuple(allowed_fps)
        self.token = secrets.token_urlsafe(32)
        fd = os.open(self.root / "access-token", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self.token + "\n")
        self.lock = threading.RLock()
        self.phase = "initializing"
        self.error = None
        self.active_id = None
        self.worker_alive = False
        self.tasks = {}
        self.queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.processes = set()
        self.cancel_requests = set()
        self.p10_registry = None

    def task_dir(self, task_id):
        if not TASK_ID.fullmatch(task_id):
            raise ValueError("Invalid task ID")
        return self.tasks_root / task_id

    def _write(self, task_id):
        destination = self.task_dir(task_id) / "status.json"
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tasks[task_id], ensure_ascii=False, indent=2) + "\n")
        tmp.replace(destination)

    def update(self, task_id, **values):
        with self.lock:
            self.tasks[task_id].update(values, updated_at=time.time())
            if values.get("status") in TERMINAL:
                self.tasks[task_id]["finished_at"] = time.time()
            self._write(task_id)

    def get(self, task_id):
        with self.lock:
            item = self.tasks.get(task_id)
            return dict(item) if item else None

    def health(self):
        with self.lock:
            return {"status": self.phase, "active_task_id": self.active_id, "error": self.error,
                    "worker_alive": self.worker_alive, "size": self.size,
                    "max_seconds": self.max_seconds, "allowed_fps": self.allowed_fps}

    def ready(self):
        with self.lock:
            self.worker_alive = True
            self.phase = "ready"

    def fail_service(self, error):
        with self.lock:
            self.error = str(error)
            self.phase = "error"

    def reserve(self):
        with self.lock:
            if self.phase == "busy" or self.active_id:
                raise AdmissionError("正在处理一个任务，请完成后再提交", 429)
            if self.phase != "ready" or not self.worker_alive or self.stop.is_set():
                raise AdmissionError("服务尚未就绪，请稍后再试", 503)
            task_id = uuid.uuid4().hex
            self.task_dir(task_id).mkdir(mode=0o700)
            self.active_id = task_id
            self.phase = "busy"
            self.tasks[task_id] = {"task_id": task_id, "status": "validating", "stage": "validating",
                                   "created_at": time.time(), "generated_chunks": 0,
                                   "stream_ready": False, "is_done": False}
            self._write(task_id)
            return task_id

    def release(self, task_id):
        with self.lock:
            self.cancel_requests.discard(task_id)
            if self.active_id == task_id:
                self.active_id = None
                if self.phase != "error":
                    self.phase = "stopped" if self.stop.is_set() else "ready"

    def register_process(self, proc):
        with self.lock:
            if self.stop.is_set():
                proc.terminate()
                raise RuntimeError("服务正在停止")
            self.processes.add(proc)

    def unregister_process(self, proc):
        with self.lock:
            self.processes.discard(proc)

    def request_cancel(self, task_id):
        with self.lock:
            if task_id in self.tasks and self.tasks[task_id]["status"] not in TERMINAL:
                self.cancel_requests.add(task_id)

    def is_cancelled(self, task_id):
        with self.lock:
            return task_id in self.cancel_requests

    def shutdown(self):
        self.stop.set()
        if self.p10_registry is not None:
            self.p10_registry.shutdown()
        with self.lock:
            self.phase = "stopped"
            for task_id, task in self.tasks.items():
                if task["status"] not in TERMINAL:
                    self.update(task_id, status="interrupted", stage="interrupted", is_done=True,
                                message="服务作业已停止")
            processes = list(self.processes)
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 5
        for proc in processes:
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
        for proc in processes:
            if proc.poll() is None:
                proc.wait(timeout=5)

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def validate_upload(state, task_id, form, files):
    try:
        fps = int(form.get("fps", 20))
    except (TypeError, ValueError):
        raise ValueError("帧率必须是整数")
    if fps not in state.allowed_fps:
        raise ValueError(f"当前支持的帧率：{state.allowed_fps}")
    prompt = str(form.get("main_prompt") or "").strip() or "一个人自然地说话"
    if len(prompt) > 2000:
        raise ValueError("提示词不能超过 2000 个字符")
    try:
        persistent_prompt = normalize_persistent_prompt(form.get("persistent_prompt"))
    except ValueError as error:
        raise ValueError(f"全程保持条件无效：{error}") from error
    try:
        edits = json.loads(form.get("prompt_json") or "[]")
    except (ValueError, TypeError):
        raise ValueError("分段提示词格式错误")
    if not isinstance(edits, list) or len(edits) > 8:
        raise ValueError("分段提示词必须是最多 8 项的列表")
    image_upload, audio_upload = files.get("img_file"), files.get("audio_file")
    if not image_upload or not audio_upload:
        raise ValueError("请同时上传图片和音频")
    run = state.task_dir(task_id)
    original_image, original_audio = run / "image.upload", run / "audio.upload"
    image_upload.save(original_image)
    audio_upload.save(original_audio)
    image_path, audio_path = run / "input.png", run / "input.wav"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(original_image) as img:
                if img.format not in {"PNG", "JPEG", "WEBP"} or img.width * img.height > 16_000_000:
                    raise ValueError("图片需为 PNG/JPEG/WebP，且不超过 1600 万像素")
                if min(img.size) < 64:
                    raise ValueError("图片宽高均需至少 64 像素")
                ImageOps.exif_transpose(img).convert("RGB").save(image_path)
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("无法读取图片") from error
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-show_entries", "format=duration,format_name:stream=codec_type",
            "-of", "json", str(original_audio),
        ], check=True, text=True, capture_output=True, timeout=15)
        info = json.loads(probe.stdout)
        seconds = float(info["format"]["duration"])
        names = set(info["format"]["format_name"].split(","))
        if not names & {"wav", "mp3", "flac", "ogg", "mov", "mp4", "m4a", "aac"}:
            raise ValueError("不支持该音频格式")
        if not any(s["codec_type"] == "audio" for s in info["streams"]):
            raise ValueError("文件中没有音轨")
        if not math.isfinite(seconds) or not 0.5 <= seconds <= state.max_seconds + 0.001:
            raise ValueError(f"音频长度需在 0.5–{state.max_seconds:g} 秒之间")
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
            "-i", str(original_audio), "-map", "0:a:0", "-vn", "-t", str(state.max_seconds + 0.1),
            "-ar", "16000", "-ac", "1", str(audio_path),
        ], check=True, capture_output=True, timeout=30)
        with wave.open(str(audio_path)) as wav:
            actual = wav.getnframes() / wav.getframerate()
        if not 0.5 <= actual <= state.max_seconds + 0.001 or abs(actual-seconds) > 0.15:
            raise ValueError("音频实际长度与文件信息不符")
        seconds = actual
    except (OSError, subprocess.SubprocessError, KeyError, json.JSONDecodeError, wave.Error) as error:
        raise ValueError("无法完整解码音频") from error
    blocks = required_blocks(target_frame_count(seconds, fps))
    try:
        normalized_edits = normalize_edits(edits, block_count=blocks, fps=fps, audio_seconds=seconds)
        conditioning_prompt(prompt, persistent_prompt=persistent_prompt)
        for edit in normalized_edits:
            conditioning_prompt(prompt, edit["prompt"], persistent_prompt)
    except ValueError as error:
        raise ValueError(f"分段提示词无效：{error}") from error
    legacy_edits = [[edit["start_block"], edit["end_block"], edit["prompt"]]
                    for edit in normalized_edits]
    params = {"task_id": task_id, "fps": fps, "main_prompt": prompt,
              "persistent_prompt": persistent_prompt,
              "edits": normalized_edits, "prompt_list": legacy_edits,
              "img_path": str(image_path), "audio_path": str(audio_path), "audio_seconds": seconds,
              "stream_with_audio": str(form.get("stream_with_audio", "true")).lower() in {"true", "1", "yes", "on"}}
    evidence = dict(params, original_image_sha256=_sha256(original_image),
                    original_audio_sha256=_sha256(original_audio),
                    prepared_audio_sha256=_sha256(audio_path),
                    requested_edits=edits, block_count=blocks,
                    effective_controls=effective_schedule(normalized_edits,
                                                         block_count=blocks, main_prompt=prompt,
                                                         persistent_prompt=persistent_prompt))
    (run / "input.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return params


def create_authenticated_app(state):
    """Shared login, same-origin checks and CSRF for video and speech services."""
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config.update(SECRET_KEY=secrets.token_hex(32), MAX_CONTENT_LENGTH=state.max_upload_mib * 1024**2,
                      MAX_FORM_MEMORY_SIZE=512 * 1024, MAX_FORM_PARTS=16,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    app.config["SERVICE_STATE"] = state

    def bearer_valid():
        supplied = request.headers.get("Authorization", "")
        return secrets.compare_digest(supplied, "Bearer " + state.token)

    @app.before_request
    def authenticate():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                return jsonify(message="请求来源不匹配"), 403
        if request.endpoint == "login":
            return None
        bearer = bearer_valid()
        if not bearer and not session.get("authenticated"):
            if request.endpoint in {"index", "p10_page", "dialogue_page"}:
                return redirect("/login")
            return jsonify(message="请先登录"), 401
        if request.method == "POST" and not bearer:
            if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""),
                                           session.get("csrf_token", "missing")):
                return jsonify(message="页面会话已变化，请刷新后重试"), 403

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Native form POSTs use Origin: null under no-referrer, which would
        # reject our own login form. Keep the origin for same-origin requests.
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = "上传内容过大" if error.code == 413 else "请求无法处理"
        return jsonify(message=message, status="error"), error.code

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            if secrets.compare_digest(request.form.get("token", ""), state.token):
                session.clear()
                session["authenticated"] = True
                session["csrf_token"] = secrets.token_urlsafe(24)
                return redirect("/")
            error = "访问口令不正确"
        return render_template("login.html", error=error), 401 if error else 200

    return app


def create_app(state):
    app = create_authenticated_app(state)

    @app.get("/")
    def index():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(24)
        return render_template("index.html", stream_resolution=state.size.replace("*", "×"),
                               max_seconds=state.max_seconds, max_upload_mib=state.max_upload_mib,
                               allowed_fps=state.allowed_fps, csrf_token=session["csrf_token"],
                               p10_enabled=app.config.get("P10_PROTOCOL", False))

    @app.get("/healthz")
    def health():
        data = state.health()
        return jsonify(data), 200 if data["status"] in {"ready", "busy"} else 503

    @app.post("/start_stream")
    def start_stream():
        task_id = None
        queued = False
        try:
            task_id = state.reserve()
            params = validate_upload(state, task_id, request.form, request.files)
            state.update(task_id, status="queued", stage="queued", message="已接受任务",
                         audio_seconds=params["audio_seconds"], fps=params["fps"],
                         stream_with_audio=params["stream_with_audio"])
            state.queue.put_nowait(params)
            queued = True
            return jsonify(status="success", task_id=task_id,
                           stream_with_audio=params["stream_with_audio"]), 202
        except AdmissionError as error:
            return jsonify(message=str(error), status="error"), error.code
        except ValueError as error:
            if task_id:
                state.update(task_id, status="failed", stage="validation", is_done=True,
                             message=str(error), error=str(error))
            return jsonify(message=str(error), status="error"), 400
        except HTTPException as error:
            if task_id:
                state.update(task_id, status="failed", stage="validation", is_done=True,
                             message="上传请求无效或过大", error=error.name)
            raise
        except Exception:
            app.logger.exception("Failed to accept task")
            if task_id:
                state.update(task_id, status="failed", stage="validation", is_done=True,
                             message="保存或验证素材失败，请检查服务日志", error="input processing failed")
            return jsonify(message="保存或验证素材失败，请检查服务日志"), 500
        finally:
            if task_id and not queued:
                state.release(task_id)

    @app.get("/task_status/<task_id>")
    def task_status(task_id):
        data = state.get(task_id)
        if data is None:
            abort(404)
        data.pop("final_video_path", None)
        data["stream_ready"] = (state.task_dir(task_id) / "hls/live.m3u8").is_file()
        if data["status"] == "finished":
            data["download_url"] = f"/download/{task_id}"
        return jsonify(data)

    @app.get("/stream/<task_id>/<path:filename>")
    def stream(task_id, filename):
        if state.get(task_id) is None or not re.fullmatch(r"live(?:\d+\.ts|\.m3u8)", filename):
            abort(404)
        mimetype = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t"
        return send_from_directory(state.task_dir(task_id) / "hls", filename, mimetype=mimetype)

    @app.get("/download/<task_id>")
    def download(task_id):
        task = state.get(task_id)
        if task is None:
            abort(404)
        if task["status"] != "finished":
            return jsonify(message="视频尚未通过验收"), 409
        path = state.task_dir(task_id) / "output.mp4"
        if not path.is_file():
            abort(404)
        return send_file(path, as_attachment=request.args.get("inline") != "1", download_name=f"liveact-{task_id}.mp4",
                         mimetype="video/mp4", conditional=True)

    return app
