"""Experimental authenticated microphone upload over same-origin WebSocket.

The accepted PCM is recorded first, then submitted to the existing asynchronous
video worker. This intentionally does not claim causal or real-time generation.
"""
from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from flask import jsonify, render_template, request, session
from starlette.middleware.wsgi import build_environ
from uvicorn import Config, Server
from werkzeug.datastructures import FileStorage

from p10_session import AudioSession, ProtocolError, SAMPLE_RATE
from video_timing import required_blocks, target_frame_count
from web_service import AdmissionError, validate_upload


class P10Registry:
    def __init__(self, state):
        self.state = state
        self.lock = threading.RLock()
        self.entries = {}

    def create(self, image_upload, form, origin):
        task_id = self.state.reserve()
        try:
            if image_upload is None:
                raise ValueError("请上传参考图片")
            task_dir = self.state.task_dir(task_id)
            image_path = task_dir / "p10-image.upload"
            image_upload.save(image_path)
            audio = AudioSession(task_dir / "p10-audio", max_seconds=self.state.max_seconds)
            with self.lock:
                self.entries[task_id] = {
                    "session": audio, "image": image_path,
                    "form": dict(form), "origin": origin.rstrip("/"),
                }
            self.state.update(task_id, status="receiving", stage="priming",
                              message="等待分块音频", p10_protocol=True)
            timer = threading.Timer(30, self.expire_unconnected, args=(task_id,))
            timer.daemon = True
            timer.start()
            return task_id, audio.token
        except Exception:
            self.state.update(task_id, status="failed", stage="validation",
                              is_done=True, message="创建录音会话失败")
            self.state.release(task_id)
            raise

    def get(self, task_id):
        with self.lock:
            return self.entries.get(task_id)

    def interrupt(self, task_id, message):
        entry = self.get(task_id)
        if entry is None:
            return
        audio = entry["session"]
        if audio.state in {"finished", "failed"}:
            return
        if audio.state != "draining":
            audio.cancel()
            self.state.update(task_id, status="interrupted", stage="interrupted",
                              is_done=True, message=message)
            self.state.release(task_id)
        else:
            self.state.request_cancel(task_id)
            self.state.update(task_id, message="已请求取消，正在安全位置停止")

    def expire_disconnect(self, task_id):
        entry = self.get(task_id)
        if entry is None:
            return
        audio = entry["session"]
        if audio.state == "interrupted" and audio.resume_deadline is not None:
            if audio.clock() >= audio.resume_deadline:
                audio.fail("Reconnect timeout")
                self.state.update(task_id, status="interrupted", stage="interrupted",
                                  is_done=True, message="重连超时，录音已中断")
                self.state.release(task_id)

    def expire_unconnected(self, task_id):
        entry = self.get(task_id)
        if entry is not None and entry["session"].state == "created":
            self.interrupt(task_id, "30 秒内未建立音频连接")

    def dispatch(self, task_id):
        entry = self.get(task_id)
        if entry is None:
            raise ValueError("Unknown P10 task")
        audio = entry["session"]
        blocks = required_blocks(target_frame_count(
            audio.received_samples / SAMPLE_RATE, 20))
        edits = [
            {"start_block": item["effective_from_block"],
             "end_block": blocks - 1, "prompt": item["prompt"], "type": item["kind"]}
            for item in audio.controls if item["effective_from_block"] < blocks
        ]
        form = dict(entry["form"], prompt_json=json.dumps(edits, ensure_ascii=False))
        try:
            with entry["image"].open("rb") as image_stream, (
                    audio.root / "input.wav").open("rb") as audio_stream:
                files = {
                    "img_file": FileStorage(image_stream, filename="image.png"),
                    "audio_file": FileStorage(audio_stream, filename="audio.wav"),
                }
                params = validate_upload(self.state, task_id, form, files)
            self.state.update(task_id, status="queued", stage="queued",
                              message="录音完成，已进入异步生成队列",
                              audio_seconds=params["audio_seconds"],
                              fps=params["fps"],
                              total_chunks=blocks, p10_control_version=audio.control_version)
            self.state.queue.put_nowait(params)
            threading.Thread(target=self._watch_output, args=(task_id,),
                             daemon=True).start()
        except Exception as error:
            audio.fail(str(error))
            self.state.update(task_id, status="failed", stage="validation",
                              is_done=True, message=f"录音验证失败：{error}")
            self.state.release(task_id)
            raise

    def _watch_output(self, task_id):
        audio = self.get(task_id)["session"]
        while not self.state.stop.is_set():
            task = self.state.get(task_id)
            if task is None:
                break
            if task["status"] == "finished":
                if audio.state == "draining":
                    audio.finish_output()
                return
            if task["status"] in {"failed", "interrupted"}:
                audio.fail(task.get("message", "Generation failed"))
                return
            time.sleep(0.5)

    def shutdown(self):
        with self.lock:
            entries = list(self.entries.values())
        for entry in entries:
            if entry["session"].state not in {"finished", "failed"}:
                entry["session"].fail("Service stopping")


def install_p10_http(app, state, registry):
    app.config["P10_PROTOCOL"] = True

    @app.get("/p10")
    def p10_page():
        if "csrf_token" not in session:
            from secrets import token_urlsafe
            session["csrf_token"] = token_urlsafe(24)
        return render_template("p10.html", csrf_token=session["csrf_token"],
                               max_seconds=state.max_seconds)

    @app.post("/p10/session")
    def create_p10_session():
        try:
            task_id, token = registry.create(request.files.get("img_file"),
                                             request.form, request.host_url)
        except AdmissionError as error:
            return jsonify(message=str(error)), error.code
        except ValueError as error:
            return jsonify(message=str(error)), 400
        return jsonify(task_id=task_id, ws_token=token,
                       ws_path=f"/p10/ws/{task_id}",
                       mode="record_then_generate"), 201

    @app.post("/p10/cancel/<task_id>")
    def cancel_p10_session(task_id):
        if registry.get(task_id) is None:
            return jsonify(message="会话不存在"), 404
        registry.interrupt(task_id, "用户取消")
        return jsonify(status="accepted"), 202

    @app.post("/p10/metrics/<task_id>")
    def save_p10_metrics(task_id):
        if registry.get(task_id) is None:
            return jsonify(message="会话不存在"), 404
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("sequence"), int):
            return jsonify(message="测量格式无效"), 400
        sequence = payload["sequence"]
        if isinstance(sequence, bool) or not 0 <= sequence <= 100000:
            return jsonify(message="测量序号无效"), 400
        mode = payload.get("mode", "preview")
        if mode not in ("preview", "full_mp4"):
            return jsonify(message="测量模式无效"), 400
        numeric = (
            "stop_to_queued_s", "stop_to_first_segment_s", "stop_to_first_frame_s",
            "play_start_s", "click_to_first_frame_s", "media_seconds", "wall_seconds", "effective_rate",
            "stall_seconds", "longest_stall_seconds", "pause_seconds",
        )
        counts = ("stall_count", "background_count", "generated_chunks")
        record = {"sequence": sequence, "task_id": task_id, "mode": mode,
                  "received_at": time.time(), "complete": payload.get("complete") is True}
        for key in numeric:
            value = payload.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                        not math.isfinite(value) or not 0 <= value <= 86400:
                    return jsonify(message=f"{key} 无效"), 400
                record[key] = float(value)
        for key in counts:
            value = payload.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100000:
                    return jsonify(message=f"{key} 无效"), 400
                record[key] = value
        if isinstance(payload.get("frame_source"), str) and payload["frame_source"] in {
                "requestVideoFrameCallback", "loadeddata"}:
            record["frame_source"] = payload["frame_source"]
        filename = "browser-metrics-mp4.json" if mode == "full_mp4" else "browser-metrics.json"
        destination = state.task_dir(task_id) / filename
        with registry.lock:
            if destination.exists():
                previous = json.loads(destination.read_text())
                if previous["sequence"] >= sequence:
                    return jsonify(status="stale"), 200
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(destination)
        return jsonify(status="saved"), 200


def build_asgi_app(flask_app, registry):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.websocket("/p10/ws/{task_id}")
    async def audio_socket(websocket: WebSocket, task_id: str):
        entry = registry.get(task_id)
        origin = websocket.headers.get("origin", "").rstrip("/")
        protocols = [part.strip() for part in websocket.headers.get(
            "sec-websocket-protocol", "").split(",")]
        if entry is None or origin != entry["origin"] or len(protocols) != 2:
            await websocket.close(code=1008)
            return
        if protocols[0] != "liveact-p10":
            await websocket.close(code=1008)
            return
        audio = entry["session"]
        try:
            snapshot = audio.connect(protocols[1])
        except ProtocolError:
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol="liveact-p10")
        await websocket.send_json(dict(type="ready", **snapshot))
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=60)
                except asyncio.TimeoutError:
                    registry.interrupt(task_id, "连续 60 秒未收到音频或控制数据")
                    await websocket.send_json({"type": "error", "code": "idle_timeout",
                                               "message": "音频输入超时"})
                    break
                if message["type"] == "websocket.disconnect":
                    break
                try:
                    if message.get("bytes") is not None:
                        answer = audio.accept_audio(message["bytes"])
                        await websocket.send_json(dict(type="ack", **answer))
                        continue
                    event = json.loads(message.get("text") or "{}")
                    if not isinstance(event, dict):
                        raise ProtocolError("Event must be an object", "event")
                    kind = event.get("type")
                    if kind == "control":
                        answer = audio.add_control(event)
                        await websocket.send_json(dict(type="control_ack", **answer))
                    elif kind == "finish":
                        answer = audio.finish()
                        try:
                            registry.dispatch(task_id)
                        except Exception as error:
                            await websocket.send_json({"type": "error", "code": "validation",
                                                       "message": str(error)})
                            break
                        await websocket.send_json(dict(type="queued", **answer))
                        break
                    elif kind == "cancel":
                        registry.interrupt(task_id, "用户取消")
                        await websocket.send_json({"type": "cancelled"})
                        break
                    else:
                        raise ProtocolError("Unknown event type", "event")
                except (ProtocolError, ValueError) as error:
                    await websocket.send_json({
                        "type": "error", "code": getattr(error, "code", "invalid"),
                        "message": str(error),
                    })
        except WebSocketDisconnect:
            pass
        finally:
            if audio.state in {"priming", "streaming"}:
                audio.disconnect()
                timer = threading.Timer(30.1, registry.expire_disconnect,
                                        args=(task_id,))
                timer.daemon = True
                timer.start()
            try:
                await websocket.close()
            except RuntimeError:
                pass

    app.mount("/", FlaskASGIAdapter(flask_app))
    return app


class FlaskASGIAdapter:
    """Small bounded WSGI bridge for the existing Flask HTTP endpoints."""

    def __init__(self, flask_app):
        self.flask_app = flask_app

    async def __call__(self, scope, receive, send):
        body = bytearray()
        max_body = self.flask_app.config["MAX_CONTENT_LENGTH"]
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            body.extend(message.get("body", b""))
            if len(body) > max_body:
                await send({"type": "http.response.start", "status": 413,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body",
                            "body": b'{"message":"upload too large"}'})
                return
            if not message.get("more_body", False):
                break
        environ = build_environ(scope, bytes(body))
        response = {}

        def start_response(status, headers, exc_info=None):
            response["status"] = int(status.split(" ", 1)[0])
            response["headers"] = headers

        iterable = self.flask_app(environ, start_response)
        try:
            chunks = iter(iterable)
            first = next(chunks, b"")
            headers = [(key.lower().encode("ascii"), value.encode("latin-1"))
                       for key, value in response["headers"]]
            await send({"type": "http.response.start",
                        "status": response["status"], "headers": headers})
            if first:
                await send({"type": "http.response.body", "body": first,
                            "more_body": True})
            for chunk in chunks:
                await send({"type": "http.response.body", "body": chunk,
                            "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        finally:
            if hasattr(iterable, "close"):
                iterable.close()


class ASGIServerAdapter:
    """Expose the same lifecycle API demo.py uses for its WSGI server."""

    def __init__(self, host, port, app):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(128)
        self.server_port = self.socket.getsockname()[1]
        self.server = Server(Config(app, host=host, port=self.server_port,
                                    log_level="warning", ws="websockets",
                                    access_log=False))

    def serve_forever(self):
        asyncio.run(self.server.serve(sockets=[self.socket]))

    def shutdown(self):
        self.server.should_exit = True

    def server_close(self):
        self.socket.close()
