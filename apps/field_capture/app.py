"""Local-first mobile field capture and CPU inference service."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.field_capture.inference import CpuYoloDetector  # noqa: E402
from container_vision.field_capture.sessions import SessionStore  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"


class CaptureHandler(BaseHTTPRequestHandler):
    store: SessionStore
    detector: CpuYoloDetector
    max_upload_bytes: int
    auth_token: str | None

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[field-capture] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self.send_file(STATIC_DIR / "index.html")
            elif parsed.path == "/api/health":
                self.send_json(
                    {
                        "ok": True,
                        "inference_enabled": self.detector.enabled,
                        "device": "cpu",
                        "auth_required": bool(self.auth_token),
                    }
                )
            elif parsed.path == "/api/sessions":
                if not self.authorized():
                    return
                self.send_json({"sessions": self.store.list()})
            elif parsed.path.endswith("/download.zip") and parsed.path.startswith("/api/sessions/"):
                if not self.authorized():
                    return
                session_id = parsed.path.removeprefix("/api/sessions/").removesuffix(
                    "/download.zip"
                ).strip("/")
                archive_path = self.store.get(session_id).archive()
                self.send_file(
                    archive_path,
                    content_type="application/zip",
                    attachment_name=archive_path.name,
                )
            elif parsed.path.endswith("/latest-frame") and parsed.path.startswith("/api/sessions/"):
                if not self.authorized():
                    return
                session_id = parsed.path.removeprefix("/api/sessions/").removesuffix("/latest-frame").strip("/")
                session = self.store.get(session_id)
                sequence = session.latest_result.get("sequence")
                if sequence is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "No frame received yet")
                else:
                    self.send_file(
                        session.directory / "inference_frames" / f"frame_{int(sequence):08d}.jpg"
                    )
            elif parsed.path.startswith("/api/sessions/"):
                if not self.authorized():
                    return
                session_id = parsed.path.removeprefix("/api/sessions/").split("/", 1)[0]
                self.send_json(self.store.get(session_id).metadata())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError:
            self.send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if not self.authorized():
                return
            if parsed.path == "/api/sessions":
                payload = self.read_json()
                session = self.store.create(
                    logical_streams=int(payload.get("logical_streams", 2)),
                    device_name=str(payload.get("device_name", "iPhone"))[:80],
                )
                self.send_json(session.metadata(), status=HTTPStatus.CREATED)
                return

            parts = parsed.path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["api", "sessions"]:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            session = self.store.get(parts[2])
            action = parts[3]
            if action == "frames":
                self.handle_frame(session, parse_qs(parsed.query))
            elif action == "segments":
                self.handle_segment(session, parse_qs(parsed.query))
            elif action == "client-metrics":
                session.save_client_metric(self.read_json())
                self.send_json({"ok": True})
            elif action == "stop":
                session.stop()
                self.send_json(session.metadata())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError:
            self.send_json({"error": "Session not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def handle_frame(self, session, query: dict) -> None:
        started = time.perf_counter()
        receive_time_ms = time.time() * 1000
        sequence = int(query.get("sequence", ["0"])[0])
        capture_time_ms = float(query.get("capture_time_ms", ["0"])[0])
        client_skipped = int(query.get("client_skipped", ["0"])[0])
        jpeg = self.read_body()
        detections, decode_ms, inference_ms = self.detector.predict(jpeg, session.logical_streams)
        server_total_ms = (time.perf_counter() - started) * 1000
        result = {
            "session_id": session.session_id,
            "sequence": sequence,
            "detections": detections,
            "metrics": {
                "sequence": sequence,
                "capture_time_ms": round(capture_time_ms, 3),
                "server_receive_time_ms": round(receive_time_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "server_total_ms": round(server_total_ms, 3),
                "logical_streams": session.logical_streams,
                "detections": len(detections),
                "jpeg_bytes": len(jpeg),
                "client_skipped": client_skipped,
            },
        }
        session.save_frame_result(sequence, jpeg, result)
        self.send_json(result)

    def handle_segment(self, session, query: dict) -> None:
        index = int(query.get("index", ["0"])[0])
        extension = query.get("extension", ["webm"])[0]
        target = session.save_segment(index, self.read_body(), extension)
        self.send_json({"ok": True, "segment": target.name})

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty request body")
        if length > self.max_upload_bytes:
            raise ValueError("Upload exceeds configured size limit")
        return self.rfile.read(length)

    def read_json(self) -> dict:
        return json.loads(self.read_body().decode("utf-8"))

    def authorized(self) -> bool:
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        bearer = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        query_token = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        provided = self.headers.get("X-Field-Capture-Token", "") or bearer or query_token
        if provided == self.auth_token:
            return True
        self.send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
        return False

    def send_file(
        self,
        path: Path,
        *,
        content_type: str | None = None,
        attachment_name: str | None = None,
    ) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        if attachment_name:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def make_handler(
    store: SessionStore,
    detector: CpuYoloDetector,
    max_upload_mb: int,
    auth_token: str | None,
):
    class ConfiguredHandler(CaptureHandler):
        pass

    ConfiguredHandler.store = store
    ConfiguredHandler.detector = detector
    ConfiguredHandler.max_upload_bytes = max_upload_mb * 1024 * 1024
    ConfiguredHandler.auth_token = auth_token
    return ConfiguredHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mobile field-capture service.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--storage-dir", default=os.environ.get("FIELD_CAPTURE_STORAGE_DIR", "data/field_captures")
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("FIELD_CAPTURE_WEIGHTS") or None,
        help="Custom YOLO weights; omit for latency-only mode",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--max-upload-mb", type=int, default=int(os.environ.get("MAX_UPLOAD_MB", "250"))
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("FIELD_CAPTURE_TOKEN") or None,
        help="Optional token required by API requests; pass it to the UI with ?token=...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve() if args.weights else None
    store = SessionStore(Path(args.storage_dir))
    detector = CpuYoloDetector(weights, confidence=args.confidence, image_size=args.image_size)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(store, detector, args.max_upload_mb, args.auth_token),
    )
    print(f"Field capture running at http://{args.host}:{args.port}")
    print(f"Storage: {store.root}")
    print(f"Inference: {'CPU YOLO ' + str(weights) if weights else 'latency-only (no weights)'}")
    print(f"Auth: {'enabled' if args.auth_token else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping field capture.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
