"""Tiny browser annotation workbench.

This first workbench is intentionally dependency-free so it can run on a fresh
remote Linux machine with only Python installed.

Example:
    python apps/workbench/app.py --images-dir data/frames \
      --annotations-dir data/annotations/manual \
      --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.data import (  # noqa: E402
    AnnotationObject,
    BoundingBox,
    DETECTION_CLASSES,
    ImageAnnotation,
    PAINTED_NUMBER_CLASS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}


@dataclass(frozen=True)
class WorkbenchConfig:
    images_dir: Path
    annotations_dir: Path
    session_id: str | None
    host: str
    port: int
    predictions_dir: Path | None = None
    benchmarks_dir: Path | None = None
    field_captures_dir: Path | None = None


def safe_relative_path(root: Path, relative_path: str) -> Path:
    """Resolve a browser-provided relative path without allowing traversal."""
    normalized = unquote(relative_path).replace("\\", "/").lstrip("/")
    candidate = (root / normalized).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("Path escapes configured directory")
    return candidate


def annotation_path_for(annotations_dir: Path, image_id: str) -> Path:
    return safe_relative_path(annotations_dir, f"{image_id}.json")


def path_has_images(path: Path) -> bool:
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in path.rglob("*"))


def list_sessions(images_dir: Path) -> list[dict[str, str | int]]:
    sessions = []
    for path in sorted(images_dir.iterdir() if images_dir.exists() else []):
        if not path.is_dir() or not path_has_images(path):
            continue
        session_id = path.name
        sessions.append(
            {
                "id": session_id,
                "name": session_id,
                "image_count": len(list_images(images_dir, session_id=session_id)),
            }
        )

    if not sessions and path_has_images(images_dir):
        sessions.append(
            {
                "id": "",
                "name": images_dir.name,
                "image_count": len(list_images(images_dir)),
            }
        )
    return sessions


def session_images_dir(images_dir: Path, session_id: str | None) -> Path:
    if session_id:
        return safe_relative_path(images_dir, session_id)
    return images_dir.resolve()


def image_id_for(*, session_id: str | None, image_path: Path, root: Path) -> str:
    relative_path = image_path.relative_to(root).as_posix()
    if session_id:
        return f"{session_id}/{relative_path}"
    return relative_path


def list_images(images_dir: Path, *, session_id: str | None = None) -> list[dict[str, str]]:
    images = []
    root = session_images_dir(images_dir, session_id)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_id = image_id_for(session_id=session_id, image_path=path, root=root)
            images.append(
                {
                    "id": image_id,
                    "name": path.name,
                    "path": path.relative_to(root).as_posix(),
                    "url": f"/images/{image_id}",
                }
            )
    return images


def list_prediction_videos(predictions_dir: Path) -> list[dict[str, str]]:
    videos = []
    if not predictions_dir.exists():
        return videos
    for path in sorted(predictions_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(predictions_dir).as_posix()
        parts = Path(relative).parts
        videos.append(
            {
                "id": relative,
                "name": path.name,
                "session": parts[0] if len(parts) > 1 else "predictions",
                "url": f"/prediction-media/{relative}",
            }
        )
    return videos


def list_benchmark_matrices(benchmarks_dir: Path) -> list[dict]:
    matrices = []
    if not benchmarks_dir.exists():
        return matrices
    for matrix_path in sorted(benchmarks_dir.glob("*/matrix.json"), reverse=True):
        matrix_dir = matrix_path.parent
        try:
            matrix = json.loads(matrix_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        runs = []
        for run_dir in sorted(path for path in matrix_dir.iterdir() if path.is_dir()):
            live_path = run_dir / "live.json"
            status_path = run_dir / "status.json"
            if not live_path.exists() and not status_path.exists():
                continue
            payload = {}
            for candidate in (live_path, status_path):
                if candidate.exists():
                    try:
                        payload = json.loads(candidate.read_text())
                        break
                    except (OSError, json.JSONDecodeError):
                        pass
            plots = []
            plots_dir = run_dir / "plots"
            if plots_dir.exists():
                plots = [
                    f"/benchmark-media/{matrix_dir.name}/{run_dir.name}/plots/{path.name}"
                    for path in sorted(plots_dir.glob("*.png"))
                ]
            runs.append({"name": run_dir.name, "live": payload, "plots": plots})
        comparison = [
            f"/benchmark-media/{matrix_dir.name}/comparison_plots/{path.name}"
            for path in sorted((matrix_dir / "comparison_plots").glob("*.png"))
        ] if (matrix_dir / "comparison_plots").exists() else []
        matrices.append({"id": matrix_dir.name, "matrix": matrix, "runs": runs, "comparison_plots": comparison})
    return matrices


def numeric(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "average": None, "p95": None, "p99": None, "min": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def list_field_capture_sessions(field_captures_dir: Path) -> list[dict]:
    sessions = []
    if not field_captures_dir.exists():
        return sessions
    for metadata_path in sorted(field_captures_dir.glob("*/session.json"), reverse=True):
        session_dir = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        frame_rows = read_csv_rows(session_dir / "metrics" / "frames.csv")
        client_rows = read_csv_rows(session_dir / "metrics" / "client.csv")
        client_by_sequence = {
            str(row.get("sequence", "")): row
            for row in client_rows
            if str(row.get("sequence", "")) != ""
        }
        points = []
        for row in frame_rows:
            sequence = str(row.get("sequence", ""))
            client = client_by_sequence.get(sequence, {})
            points.append(
                {
                    "sequence": numeric(sequence),
                    "server_total_ms": numeric(row.get("server_total_ms")),
                    "decode_ms": numeric(row.get("decode_ms")),
                    "inference_ms": numeric(row.get("inference_ms")),
                    "round_trip_ms": numeric(client.get("round_trip_ms")),
                    "jpeg_bytes": numeric(row.get("jpeg_bytes")),
                    "client_skipped": numeric(row.get("client_skipped")),
                    "detections": numeric(row.get("detections")),
                }
            )

        videos = []
        for path in sorted((session_dir / "cloud_recording").glob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                relative = path.relative_to(field_captures_dir).as_posix()
                videos.append(
                    {
                        "id": relative,
                        "name": path.name,
                        "url": f"/field-capture-media/{relative}",
                        "bytes": path.stat().st_size,
                    }
                )

        frames = []
        for path in sorted((session_dir / "inference_frames").glob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                relative = path.relative_to(field_captures_dir).as_posix()
                frames.append(
                    {
                        "id": relative,
                        "name": path.name,
                        "url": f"/field-capture-media/{relative}",
                    }
                )

        summaries = {
            "server_total_ms": summarize_values(
                [value for value in (point["server_total_ms"] for point in points) if value is not None]
            ),
            "round_trip_ms": summarize_values(
                [value for value in (point["round_trip_ms"] for point in points) if value is not None]
            ),
            "decode_ms": summarize_values(
                [value for value in (point["decode_ms"] for point in points) if value is not None]
            ),
            "inference_ms": summarize_values(
                [value for value in (point["inference_ms"] for point in points) if value is not None]
            ),
            "jpeg_bytes": summarize_values(
                [value for value in (point["jpeg_bytes"] for point in points) if value is not None]
            ),
        }
        sessions.append(
            {
                "id": metadata.get("session_id") or session_dir.name,
                "name": session_dir.name,
                "metadata": metadata,
                "videos": videos,
                "frames": frames,
                "points": points,
                "summaries": summaries,
                "metrics": {
                    "frame_rows": len(frame_rows),
                    "client_rows": len(client_rows),
                    "total_client_skipped": sum(
                        value
                        for value in (point["client_skipped"] for point in points)
                        if value is not None
                    ),
                    "total_frame_detections": sum(
                        value for value in (point["detections"] for point in points) if value is not None
                    ),
                },
            }
        )
    return sessions


def session_id_from_image_id(config: WorkbenchConfig, image_id: str) -> str:
    if config.session_id:
        return config.session_id
    parts = image_id.split("/", maxsplit=1)
    if len(parts) == 2:
        return parts[0]
    return config.images_dir.name or "manual_session"


def annotation_from_payload(config: WorkbenchConfig, payload: dict) -> ImageAnnotation:
    image_id = str(payload["image_id"])
    width = int(payload["width"])
    height = int(payload["height"])
    objects = []
    for index, raw_object in enumerate(payload.get("objects", []), start=1):
        readable = bool(raw_object.get("readable", True))
        transcription = str(raw_object.get("transcription", "")).strip()
        objects.append(
            AnnotationObject(
                object_id=str(raw_object.get("object_id") or f"box_{index:03d}"),
                class_name=str(raw_object.get("class_name", PAINTED_NUMBER_CLASS)),
                bbox=BoundingBox.from_xyxy(raw_object["bbox_xyxy"]),
                transcription=transcription if readable else None,
                readable=readable,
                occluded=bool(raw_object.get("occluded", False)),
            )
        )
    return ImageAnnotation(
        image_id=image_id,
        session_id=session_id_from_image_id(config, image_id),
        file=image_id,
        width=width,
        height=height,
        objects=tuple(objects),
    )


class WorkbenchHandler(BaseHTTPRequestHandler):
    config: WorkbenchConfig

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[workbench] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            elif parsed.path == "/api/config":
                self.send_json(
                    {
                        "images_dir": str(self.config.images_dir),
                        "annotations_dir": str(self.config.annotations_dir),
                        "session_id": self.config.session_id,
                        "image_extensions": sorted(IMAGE_EXTENSIONS),
                        "detection_classes": list(DETECTION_CLASSES),
                        "predictions_dir": str(self.config.predictions_dir or ""),
                        "benchmarks_dir": str(self.config.benchmarks_dir or ""),
                        "field_captures_dir": str(self.config.field_captures_dir or ""),
                    }
                )
            elif parsed.path == "/api/sessions":
                self.send_json({"sessions": list_sessions(self.config.images_dir)})
            elif parsed.path == "/api/images":
                query = parse_qs(parsed.query)
                session_id = query.get("session", [self.config.session_id or ""])[0] or None
                self.send_json({"images": list_images(self.config.images_dir, session_id=session_id)})
            elif parsed.path == "/api/predictions":
                predictions_dir = self.config.predictions_dir or REPO_ROOT / "outputs" / "predictions"
                self.send_json({"videos": list_prediction_videos(predictions_dir)})
            elif parsed.path == "/api/benchmarks":
                benchmarks_dir = self.config.benchmarks_dir or REPO_ROOT / "runs" / "benchmarks"
                self.send_json({"matrices": list_benchmark_matrices(benchmarks_dir)})
            elif parsed.path == "/api/field-captures":
                field_captures_dir = self.config.field_captures_dir or REPO_ROOT / "data" / "field_captures"
                self.send_json({"sessions": list_field_capture_sessions(field_captures_dir)})
            elif parsed.path == "/api/annotation":
                query = parse_qs(parsed.query)
                image_id = query.get("image", [""])[0]
                self.handle_get_annotation(image_id)
            elif parsed.path.startswith("/images/"):
                self.handle_get_image(parsed.path.removeprefix("/images/"))
            elif parsed.path.startswith("/prediction-media/"):
                self.handle_get_prediction(parsed.path.removeprefix("/prediction-media/"))
            elif parsed.path.startswith("/benchmark-media/"):
                self.handle_get_benchmark_media(parsed.path.removeprefix("/benchmark-media/"))
            elif parsed.path.startswith("/field-capture-media/"):
                self.handle_get_field_capture_media(
                    parsed.path.removeprefix("/field-capture-media/")
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # pragma: no cover - safety net for UI errors
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            annotation = annotation_from_payload(self.config, payload)
            target = annotation_path_for(self.config.annotations_dir, annotation.image_id)
            annotation.save(target)
            self.send_json({"ok": True, "path": str(target)})
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def handle_get_annotation(self, image_id: str) -> None:
        if not image_id:
            self.send_json({"error": "Missing image query parameter"}, status=HTTPStatus.BAD_REQUEST)
            return
        target = annotation_path_for(self.config.annotations_dir, image_id)
        if target.exists():
            self.send_json(json.loads(target.read_text()))
        else:
            self.send_json({"image_id": image_id, "objects": []})

    def handle_get_image(self, relative_path: str) -> None:
        image_path = safe_relative_path(self.config.images_dir, relative_path)
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        content = image_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_get_prediction(self, relative_path: str) -> None:
        predictions_dir = self.config.predictions_dir or REPO_ROOT / "outputs" / "predictions"
        media_path = safe_relative_path(predictions_dir, relative_path)
        if not media_path.is_file() or media_path.suffix.lower() not in VIDEO_EXTENSIONS:
            self.send_error(HTTPStatus.NOT_FOUND, "Prediction video not found")
            return
        self.send_file(media_path)

    def handle_get_benchmark_media(self, relative_path: str) -> None:
        benchmarks_dir = self.config.benchmarks_dir or REPO_ROOT / "runs" / "benchmarks"
        media_path = safe_relative_path(benchmarks_dir, relative_path)
        if not media_path.is_file() or media_path.suffix.lower() not in {".png", ".json", ".csv", ".log"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Benchmark artifact not found")
            return
        self.send_file(media_path)

    def handle_get_field_capture_media(self, relative_path: str) -> None:
        field_captures_dir = self.config.field_captures_dir or REPO_ROOT / "data" / "field_captures"
        media_path = safe_relative_path(field_captures_dir, relative_path)
        allowed = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | {".json", ".csv", ".zip"}
        if not media_path.is_file() or media_path.suffix.lower() not in allowed:
            self.send_error(HTTPStatus.NOT_FOUND, "Field capture artifact not found")
            return
        self.send_file(media_path)

    def send_file(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            requested = range_header.removeprefix("bytes=").split(",", maxsplit=1)[0]
            start_text, end_text = requested.split("-", maxsplit=1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
            end = min(end, size - 1)
            if start < 0 or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_text(json.dumps(payload), status=status, content_type="application/json")

    def send_text(
        self,
        payload: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def make_handler(config: WorkbenchConfig) -> type[WorkbenchHandler]:
    class ConfiguredWorkbenchHandler(WorkbenchHandler):
        pass

    ConfiguredWorkbenchHandler.config = config
    return ConfiguredWorkbenchHandler


def run_server(config: WorkbenchConfig, *, open_browser: bool) -> None:
    config.images_dir.mkdir(parents=True, exist_ok=True)
    config.annotations_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    url = f"http://{config.host}:{config.port}"
    print(f"Container ID workbench running at {url}")
    print(f"Images:      {config.images_dir}")
    print(f"Annotations: {config.annotations_dir}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workbench.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Container ID annotation workbench.")
    parser.add_argument(
        "--images-dir",
        default="data/frames",
        help="Frames root or a single image-session folder to label.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="data/annotations/manual",
        help="Folder where annotation JSON files will be saved.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional: force a single capture/session id instead of selecting sessions in the UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 on a remote server.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--predictions-dir",
        default="outputs/predictions",
        help="Folder containing YOLO prediction videos shown in the UI.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="runs/benchmarks",
        help="Folder containing benchmark matrices shown in the UI.",
    )
    parser.add_argument(
        "--field-captures-dir",
        default="data/field_captures",
        help="Folder containing extracted field-capture ZIP sessions shown in the UI.",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open a local browser tab.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    config = WorkbenchConfig(
        images_dir=images_dir,
        annotations_dir=Path(args.annotations_dir).resolve(),
        session_id=args.session_id,
        host=args.host,
        port=args.port,
        predictions_dir=Path(args.predictions_dir).resolve(),
        benchmarks_dir=Path(args.benchmarks_dir).resolve(),
        field_captures_dir=Path(args.field_captures_dir).resolve(),
    )
    run_server(config, open_browser=args.open_browser)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Container ID Workbench</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #101318; color: #eef2f7; }
    header { padding: 14px 20px; border-bottom: 1px solid #293241; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
    header h1 { font-size: 18px; margin: 0; }
    main { display: grid; grid-template-columns: 280px 1fr 320px; height: calc(100vh - 58px); }
    aside, section { min-height: 0; overflow: auto; }
    aside { border-right: 1px solid #293241; padding: 12px; }
    .right { border-left: 1px solid #293241; padding: 12px; }
    button, input, select { border: 1px solid #3a4658; border-radius: 8px; padding: 8px 10px; background: #151a22; color: #eef2f7; }
    button { cursor: pointer; }
    button:hover { background: #1f2733; }
    .image-item { width: 100%; text-align: left; margin-bottom: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .image-item.active { outline: 2px solid #5aa9ff; }
    .session-item { width: 100%; text-align: left; margin-bottom: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-item.active { outline: 2px solid #8ee59b; }
    .stage { display: grid; place-items: center; padding: 20px; overflow: auto; }
    .canvas-wrap { position: relative; width: fit-content; height: fit-content; }
    canvas { position: absolute; left: 0; top: 0; cursor: crosshair; }
    #photo { display: block; max-width: calc(100vw - 650px); max-height: calc(100vh - 120px); object-fit: contain; }
    .box-row { border: 1px solid #293241; border-radius: 10px; padding: 10px; margin-bottom: 10px; background: #151a22; }
    .box-row label { display: block; font-size: 12px; color: #aab4c3; margin: 8px 0 4px; }
    .box-row input[type="text"] { width: calc(100% - 22px); }
    .muted { color: #aab4c3; font-size: 13px; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    #status { color: #8ee59b; }
    .view-tabs { display: flex; gap: 6px; margin-left: auto; }
    .view-tab.active { outline: 2px solid #5aa9ff; }
    .hidden { display: none !important; }
    #predictionView { grid-template-columns: 320px 1fr; }
    #predictionView .prediction-sidebar { border-right: 1px solid #293241; padding: 12px; overflow: auto; }
    #predictionView .prediction-stage { display: grid; place-items: center; padding: 20px; min-width: 0; }
    #predictionVideo { width: 100%; max-width: 1200px; max-height: calc(100vh - 140px); background: #000; }
    .prediction-item { width: 100%; text-align: left; margin-bottom: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .prediction-item.active { outline: 2px solid #ffb45e; }
    #benchmarkView { grid-template-columns: 320px 1fr; }
    #fieldCaptureView { grid-template-columns: 340px 1fr; }
    .benchmark-sidebar { border-right: 1px solid #293241; padding: 12px; overflow: auto; }
    .field-sidebar { border-right: 1px solid #293241; padding: 12px; overflow: auto; }
    .benchmark-content { padding: 18px; overflow: auto; }
    .field-content { padding: 18px; overflow: auto; }
    .benchmark-item { width: 100%; text-align: left; margin-bottom: 6px; }
    .benchmark-item.active { outline: 2px solid #8ee59b; }
    .field-item { width: 100%; text-align: left; margin-bottom: 6px; }
    .field-item.active { outline: 2px solid #c084fc; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; margin: 12px 0; }
    .metric-card { border: 1px solid #293241; border-radius: 10px; background: #151a22; padding: 12px; }
    .metric-card strong { display: block; font-size: 20px; margin-top: 4px; }
    .run-card { border: 1px solid #293241; border-radius: 10px; padding: 12px; margin: 12px 0; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .plot-grid img { width: 100%; background: white; border-radius: 8px; }
    .chart { width: 100%; height: 250px; background: #151a22; border: 1px solid #293241; border-radius: 10px; margin: 12px 0; }
    .chart svg { width: 100%; height: 100%; display: block; }
    .media-grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 14px; align-items: start; }
    .media-grid video, .media-grid img { width: 100%; border-radius: 10px; background: #000; }
    .segment-list { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
    .segment-list button.active { outline: 2px solid #c084fc; }
  </style>
</head>
<body>
  <header>
    <h1>Container ID Workbench</h1>
    <span class="muted">Label painted numbers and license plates, enter their text, save JSON.</span>
    <span id="status"></span>
    <nav class="view-tabs">
      <button class="view-tab active" data-view="annotationView">Annotate</button>
      <button class="view-tab" data-view="predictionView">YOLO predictions</button>
      <button class="view-tab" data-view="benchmarkView">Benchmarks</button>
      <button class="view-tab" data-view="fieldCaptureView">Field captures</button>
    </nav>
  </header>
  <main id="annotationView">
    <aside>
      <div class="toolbar">
        <button id="refresh">Refresh</button>
      </div>
      <p class="muted" id="configInfo">Loading folder…</p>
      <h3>Sessions</h3>
      <div id="sessionList"></div>
      <h3>Images</h3>
      <p class="muted" id="imageCount">Loading images…</p>
      <div id="imageList"></div>
    </aside>
    <section class="stage">
      <div class="canvas-wrap">
        <img id="photo" alt="Selected image" />
        <canvas id="overlay"></canvas>
      </div>
    </section>
    <aside class="right">
      <div class="toolbar">
        <button id="save">Save</button>
        <button id="undo">Undo box</button>
        <button id="clear">Clear</button>
      </div>
      <label class="muted" for="newBoxClass">Class for new boxes</label>
      <select id="newBoxClass">
        <option value="painted_number">painted_number</option>
        <option value="license_plate">license_plate</option>
      </select>
      <p class="muted">Tip: choose a class, then drag on the image. Use one box per visible identifier.</p>
      <div id="boxList"></div>
    </aside>
  </main>
  <main id="predictionView" class="hidden">
    <aside class="prediction-sidebar">
      <div class="toolbar">
        <button id="refreshPredictions">Refresh predictions</button>
      </div>
      <p class="muted" id="predictionFolder">Loading prediction folder…</p>
      <p class="muted" id="predictionCount"></p>
      <div id="predictionList"></div>
    </aside>
    <section class="prediction-stage">
      <div>
        <h3 id="predictionTitle">Select a prediction video</h3>
        <video id="predictionVideo" controls playsinline preload="metadata"></video>
        <p class="muted">Boxes, classes and confidence values are embedded in this video.</p>
      </div>
    </section>
  </main>
  <main id="benchmarkView" class="hidden">
    <aside class="benchmark-sidebar">
      <div class="toolbar">
        <button id="refreshBenchmarks">Refresh benchmarks</button>
        <button id="toggleBenchmarkPolling">Pause live updates</button>
      </div>
      <p class="muted" id="benchmarkFolder">Loading benchmark folder…</p>
      <div id="benchmarkList"></div>
    </aside>
    <section class="benchmark-content">
      <h2 id="benchmarkTitle">No benchmark selected</h2>
      <p class="muted" id="benchmarkStatus"></p>
      <div id="benchmarkRuns"></div>
      <h2>CPU vs GPU comparisons</h2>
      <div class="plot-grid" id="comparisonPlots"></div>
    </section>
  </main>
  <main id="fieldCaptureView" class="hidden">
    <aside class="field-sidebar">
      <div class="toolbar">
        <button id="refreshFieldCaptures">Refresh captures</button>
      </div>
      <p class="muted" id="fieldCaptureFolder">Loading field-capture folder…</p>
      <p class="muted" id="fieldCaptureCount"></p>
      <div id="fieldCaptureList"></div>
    </aside>
    <section class="field-content">
      <h2 id="fieldCaptureTitle">No field capture selected</h2>
      <p class="muted" id="fieldCaptureStatus"></p>
      <div id="fieldCaptureSummary" class="metric-grid"></div>
      <div class="media-grid">
        <div>
          <h3>Recorded video segments</h3>
          <div id="fieldSegmentList" class="segment-list"></div>
          <video id="fieldCaptureVideo" controls playsinline preload="metadata"></video>
        </div>
        <div>
          <h3>Sampled inference frame</h3>
          <img id="fieldCaptureFrame" alt="Latest sampled frame">
          <p class="muted">These are the JPEG frames sent to the backend for timing/inference.</p>
        </div>
      </div>
      <h3>Latency over time</h3>
      <div id="fieldLatencyChart" class="chart"></div>
      <h3>Pipeline stages</h3>
      <div id="fieldStageChart" class="chart"></div>
    </section>
  </main>
<script>
const imageList = document.getElementById("imageList");
const sessionList = document.getElementById("sessionList");
const imageCount = document.getElementById("imageCount");
const configInfo = document.getElementById("configInfo");
const photo = document.getElementById("photo");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const boxList = document.getElementById("boxList");
const statusEl = document.getElementById("status");
const newBoxClass = document.getElementById("newBoxClass");
const predictionList = document.getElementById("predictionList");
const predictionCount = document.getElementById("predictionCount");
const predictionFolder = document.getElementById("predictionFolder");
const predictionVideo = document.getElementById("predictionVideo");
const predictionTitle = document.getElementById("predictionTitle");
const benchmarkFolder = document.getElementById("benchmarkFolder");
const benchmarkList = document.getElementById("benchmarkList");
const benchmarkTitle = document.getElementById("benchmarkTitle");
const benchmarkStatus = document.getElementById("benchmarkStatus");
const benchmarkRuns = document.getElementById("benchmarkRuns");
const comparisonPlots = document.getElementById("comparisonPlots");
const benchmarkContent = document.querySelector(".benchmark-content");
const toggleBenchmarkPolling = document.getElementById("toggleBenchmarkPolling");
const fieldCaptureFolder = document.getElementById("fieldCaptureFolder");
const fieldCaptureCount = document.getElementById("fieldCaptureCount");
const fieldCaptureList = document.getElementById("fieldCaptureList");
const fieldCaptureTitle = document.getElementById("fieldCaptureTitle");
const fieldCaptureStatus = document.getElementById("fieldCaptureStatus");
const fieldCaptureSummary = document.getElementById("fieldCaptureSummary");
const fieldSegmentList = document.getElementById("fieldSegmentList");
const fieldCaptureVideo = document.getElementById("fieldCaptureVideo");
const fieldCaptureFrame = document.getElementById("fieldCaptureFrame");
const fieldLatencyChart = document.getElementById("fieldLatencyChart");
const fieldStageChart = document.getElementById("fieldStageChart");

let images = [];
let sessions = [];
let activeSession = null;
let activeImage = null;
let boxes = [];
let drawing = null;
let predictionVideos = [];
let activePrediction = null;
let benchmarkMatrices = [];
let activeBenchmark = null;
let benchmarkPoll = null;
let benchmarkPollingEnabled = true;
let fieldCaptures = [];
let activeFieldCapture = null;
let activeFieldSegment = null;

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  configInfo.innerHTML = `
    Reading images from:<br>
    <code>${config.images_dir}</code><br><br>
    Saving annotations to:<br>
    <code>${config.annotations_dir}</code><br><br>
    Supported: ${config.image_extensions.join(", ")}
  `;
  predictionFolder.innerHTML = `Reading predictions from:<br><code>${config.predictions_dir}</code>`;
  benchmarkFolder.innerHTML = `Reading benchmarks from:<br><code>${config.benchmarks_dir}</code>`;
  fieldCaptureFolder.innerHTML = `Reading field captures from:<br><code>${config.field_captures_dir}</code>`;
  if (config.detection_classes) {
    newBoxClass.innerHTML = config.detection_classes
      .map(name => `<option value="${name}">${name}</option>`)
      .join("");
  }
}

function selectView(viewId) {
  document.querySelectorAll("main").forEach(view => view.classList.toggle("hidden", view.id !== viewId));
  document.querySelectorAll(".view-tab").forEach(tab => tab.classList.toggle("active", tab.dataset.view === viewId));
  if (viewId === "predictionView") loadPredictions().catch(error => setStatus(error.message, false));
  if (viewId === "fieldCaptureView") loadFieldCaptures().catch(error => setStatus(error.message, false));
  if (benchmarkPoll) clearInterval(benchmarkPoll);
  if (viewId === "benchmarkView") {
    loadBenchmarks().catch(error => setStatus(error.message, false));
    startBenchmarkPolling();
  }
}

function startBenchmarkPolling() {
  if (benchmarkPoll) clearInterval(benchmarkPoll);
  benchmarkPoll = null;
  if (!benchmarkPollingEnabled) return;
  benchmarkPoll = setInterval(() => loadBenchmarks(true).catch(() => {}), 5000);
}

async function loadBenchmarks(preserveSelection = false) {
  const response = await fetch("/api/benchmarks");
  const data = await response.json();
  benchmarkMatrices = data.matrices || [];
  if ((!preserveSelection || !activeBenchmark) && benchmarkMatrices.length) activeBenchmark = benchmarkMatrices[0].id;
  renderBenchmarkList();
  renderBenchmarkDetails();
  const selected = benchmarkMatrices.find(item => item.id === activeBenchmark);
  if (selected?.matrix?.status === "complete" && benchmarkPoll) {
    clearInterval(benchmarkPoll);
    benchmarkPoll = null;
  }
}

function renderBenchmarkList() {
  benchmarkList.innerHTML = "";
  if (!benchmarkMatrices.length) {
    benchmarkList.innerHTML = `<p class="muted">No benchmark matrices found yet.</p>`;
    return;
  }
  benchmarkMatrices.forEach(item => {
    const button = document.createElement("button");
    button.className = "benchmark-item" + (item.id === activeBenchmark ? " active" : "");
    button.textContent = `${item.id} — ${item.matrix.status || "unknown"}`;
    button.onclick = () => { activeBenchmark = item.id; renderBenchmarkList(); renderBenchmarkDetails(); };
    benchmarkList.appendChild(button);
  });
}

function metricCard(label, value) {
  return `<div class="metric-card"><span class="muted">${label}</span><strong>${value}</strong></div>`;
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function renderBenchmarkDetails() {
  const selected = benchmarkMatrices.find(item => item.id === activeBenchmark);
  if (!selected) return;
  const previousScroll = benchmarkContent.scrollTop;
  benchmarkTitle.textContent = selected.id;
  benchmarkStatus.textContent = `Status: ${selected.matrix.status || "unknown"} · ${selected.runs.length} run folder(s)`;
  benchmarkRuns.innerHTML = "";
  selected.runs.forEach(run => {
    const live = run.live || {};
    const latency = live.latency_ms || live.latencies_ms?.end_to_end_ms || {};
    const resource = live.latest_resource || Object.fromEntries(
      Object.entries(live.resources || {}).map(([key, value]) => [key, value.average])
    );
    const progress = live.status === "complete" ? 100 : live.progress_percent;
    const card = document.createElement("div");
    card.className = "run-card";
    card.innerHTML = `
      <h3>${run.name} — ${live.status || "initializing"}</h3>
      <div class="metric-grid">
        ${metricCard("Progress", `${formatNumber(progress, 0)}%`)}
        ${metricCard("Processed FPS", formatNumber(live.processed_fps, 2))}
        ${metricCard("Dropped frames", live.dropped_frames ?? "—")}
        ${metricCard("Queue depth", live.queue_depth ?? "—")}
        ${metricCard("Average latency", `${formatNumber(latency.average)} ms`)}
        ${metricCard("p95 latency", `${formatNumber(latency.p95)} ms`)}
        ${metricCard("CPU", `${formatNumber(resource.cpu_percent)}%`)}
        ${metricCard("Process RAM", `${formatNumber(resource.process_ram_mb)} MB`)}
        ${metricCard("System RAM", `${formatNumber(resource.ram_percent)}%`)}
        ${metricCard("CPU temperature", `${formatNumber(resource.cpu_temperature_c)} °C`)}
        ${metricCard("CPU power", `${formatNumber(resource.cpu_power_w)} W`)}
        ${metricCard("GPU", `${formatNumber(resource.gpu_util_percent)}%`)}
        ${metricCard("GPU memory", `${formatNumber(resource.gpu_memory_mb)} MB`)}
        ${metricCard("GPU temperature", `${formatNumber(resource.gpu_temperature_c)} °C`)}
        ${metricCard("GPU power", `${formatNumber(resource.gpu_power_w)} W`)}
      </div>
      <div class="plot-grid">${run.plots.map(url => `<img src="${url}" alt="Benchmark plot">`).join("")}</div>
    `;
    benchmarkRuns.appendChild(card);
  });
  comparisonPlots.innerHTML = selected.comparison_plots
    .map(url => `<img src="${url}" alt="Comparison plot">`).join("");
  benchmarkContent.scrollTop = previousScroll;
}

async function loadPredictions() {
  const response = await fetch("/api/predictions");
  const data = await response.json();
  predictionVideos = data.videos || [];
  predictionCount.textContent = `${predictionVideos.length} prediction video(s)`;
  renderPredictions();
  if (!activePrediction && predictionVideos.length) selectPrediction(predictionVideos[0].id);
}

function renderPredictions() {
  predictionList.innerHTML = "";
  if (!predictionVideos.length) {
    predictionList.innerHTML = `<p class="muted">No predicted videos found. Run <code>scripts/predict_yolo.py</code> first.</p>`;
    predictionVideo.removeAttribute("src");
    return;
  }
  let previousSession = null;
  predictionVideos.forEach(video => {
    if (video.session !== previousSession) {
      const heading = document.createElement("h3");
      heading.textContent = video.session;
      predictionList.appendChild(heading);
      previousSession = video.session;
    }
    const button = document.createElement("button");
    button.className = "prediction-item" + (activePrediction === video.id ? " active" : "");
    button.textContent = video.name;
    button.title = video.id;
    button.onclick = () => selectPrediction(video.id);
    predictionList.appendChild(button);
  });
}

function selectPrediction(videoId) {
  const selected = predictionVideos.find(video => video.id === videoId);
  if (!selected) return;
  activePrediction = selected.id;
  predictionTitle.textContent = `${selected.session} / ${selected.name}`;
  predictionVideo.src = selected.url;
  predictionVideo.load();
  renderPredictions();
}

async function loadFieldCaptures() {
  const response = await fetch("/api/field-captures");
  const data = await response.json();
  fieldCaptures = data.sessions || [];
  fieldCaptureCount.textContent = `${fieldCaptures.length} field capture session(s)`;
  if (!activeFieldCapture && fieldCaptures.length) activeFieldCapture = fieldCaptures[0].id;
  renderFieldCaptureList();
  renderFieldCaptureDetails();
}

function renderFieldCaptureList() {
  fieldCaptureList.innerHTML = "";
  if (!fieldCaptures.length) {
    fieldCaptureList.innerHTML = `
      <p class="muted">
        No extracted field captures found. Unzip each Cloud Run/session ZIP into
        <code>data/field_captures/&lt;session_id&gt;/</code>.
      </p>
    `;
    return;
  }
  fieldCaptures.forEach(item => {
    const meta = item.metadata || {};
    const button = document.createElement("button");
    button.className = "field-item" + (item.id === activeFieldCapture ? " active" : "");
    button.innerHTML = `
      <strong>${item.name}</strong><br>
      <span class="muted">${meta.status || "unknown"} · ${item.videos.length} video(s) · ${item.metrics.frame_rows} frame row(s)</span>
    `;
    button.onclick = () => {
      activeFieldCapture = item.id;
      activeFieldSegment = null;
      renderFieldCaptureList();
      renderFieldCaptureDetails();
    };
    fieldCaptureList.appendChild(button);
  });
}

function summaryValue(summary, key, digits = 1) {
  return summary?.[key] === null || summary?.[key] === undefined ? "—" : formatNumber(summary[key], digits);
}

function renderFieldCaptureDetails() {
  const selected = fieldCaptures.find(item => item.id === activeFieldCapture);
  if (!selected) {
    fieldCaptureTitle.textContent = "No field capture selected";
    fieldCaptureStatus.textContent = "";
    fieldCaptureSummary.innerHTML = "";
    fieldSegmentList.innerHTML = "";
    fieldCaptureVideo.removeAttribute("src");
    fieldCaptureFrame.removeAttribute("src");
    fieldLatencyChart.innerHTML = "";
    fieldStageChart.innerHTML = "";
    return;
  }
  const meta = selected.metadata || {};
  const roundTrip = selected.summaries.round_trip_ms;
  const server = selected.summaries.server_total_ms;
  const inference = selected.summaries.inference_ms;
  const decode = selected.summaries.decode_ms;
  fieldCaptureTitle.textContent = selected.name;
  fieldCaptureStatus.textContent = [
    `Status: ${meta.status || "unknown"}`,
    `Started: ${meta.started_at || "—"}`,
    `Logical streams: ${meta.logical_streams ?? "—"}`,
    `Rows: ${selected.metrics.frame_rows} server / ${selected.metrics.client_rows} client`,
  ].join(" · ");
  fieldCaptureSummary.innerHTML = `
    ${metricCard("Avg round trip", `${summaryValue(roundTrip, "average")} ms`)}
    ${metricCard("p95 / p99 round trip", `${summaryValue(roundTrip, "p95")} / ${summaryValue(roundTrip, "p99")} ms`)}
    ${metricCard("Avg server total", `${summaryValue(server, "average")} ms`)}
    ${metricCard("p95 / p99 server", `${summaryValue(server, "p95")} / ${summaryValue(server, "p99")} ms`)}
    ${metricCard("Avg inference", `${summaryValue(inference, "average")} ms`)}
    ${metricCard("Avg decode", `${summaryValue(decode, "average")} ms`)}
    ${metricCard("Skipped intervals", formatNumber(selected.metrics.total_client_skipped, 0))}
    ${metricCard("Detections", formatNumber(meta.detections ?? selected.metrics.total_frame_detections, 0))}
    ${metricCard("Uploaded data", `${formatNumber((meta.bytes_received || 0) / 1048576, 1)} MB`)}
  `;
  renderFieldSegments(selected);
  if (selected.frames.length) {
    fieldCaptureFrame.src = selected.frames[selected.frames.length - 1].url;
  } else {
    fieldCaptureFrame.removeAttribute("src");
  }
  renderLineChart(fieldLatencyChart, selected.points, [
    {key: "round_trip_ms", label: "round trip", color: "#c084fc"},
    {key: "server_total_ms", label: "server total", color: "#60a5fa"},
  ]);
  renderLineChart(fieldStageChart, selected.points, [
    {key: "decode_ms", label: "decode", color: "#8ee59b"},
    {key: "inference_ms", label: "inference", color: "#ffb45e"},
    {key: "server_total_ms", label: "server total", color: "#60a5fa"},
  ]);
}

function renderFieldSegments(selected) {
  fieldSegmentList.innerHTML = "";
  if (!selected.videos.length) {
    fieldSegmentList.innerHTML = `<p class="muted">No video segments found in this session.</p>`;
    fieldCaptureVideo.removeAttribute("src");
    return;
  }
  if (!activeFieldSegment || !selected.videos.find(video => video.id === activeFieldSegment)) {
    activeFieldSegment = selected.videos[0].id;
  }
  selected.videos.forEach((video, index) => {
    const button = document.createElement("button");
    button.className = activeFieldSegment === video.id ? "active" : "";
    button.textContent = `Segment ${index + 1}`;
    button.title = `${video.name} · ${(video.bytes / 1048576).toFixed(1)} MB`;
    button.onclick = () => {
      activeFieldSegment = video.id;
      renderFieldSegments(selected);
    };
    fieldSegmentList.appendChild(button);
  });
  const selectedVideo = selected.videos.find(video => video.id === activeFieldSegment);
  if (selectedVideo && fieldCaptureVideo.src !== selectedVideo.url) {
    fieldCaptureVideo.src = selectedVideo.url;
    fieldCaptureVideo.load();
  }
}

function renderLineChart(container, points, series) {
  const clean = points
    .filter(point => Number.isFinite(Number(point.sequence)))
    .map(point => ({...point, sequence: Number(point.sequence)}));
  const values = [];
  clean.forEach(point => {
    series.forEach(item => {
      const value = Number(point[item.key]);
      if (Number.isFinite(value)) values.push(value);
    });
  });
  if (!clean.length || !values.length) {
    container.innerHTML = `<p class="muted" style="padding:12px">No metric rows available for this chart.</p>`;
    return;
  }
  const width = 900;
  const height = 250;
  const pad = {left: 54, right: 20, top: 20, bottom: 34};
  const minX = Math.min(...clean.map(point => point.sequence));
  const maxX = Math.max(...clean.map(point => point.sequence));
  const minY = 0;
  const maxY = Math.max(1, ...values) * 1.08;
  const x = value => pad.left + ((value - minX) / Math.max(1, maxX - minX)) * (width - pad.left - pad.right);
  const y = value => height - pad.bottom - ((value - minY) / Math.max(1, maxY - minY)) * (height - pad.top - pad.bottom);
  const lines = series.map(item => {
    const coords = clean
      .map(point => ({x: x(point.sequence), y: y(Number(point[item.key]))}))
      .filter(point => Number.isFinite(point.y))
      .map(point => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(" ");
    return coords ? `<polyline fill="none" stroke="${item.color}" stroke-width="2" points="${coords}" />` : "";
  }).join("");
  const legend = series.map((item, index) =>
    `<g transform="translate(${pad.left + index * 150}, ${height - 10})"><rect width="10" height="10" fill="${item.color}"/><text x="16" y="10" fill="#eef2f7" font-size="12">${item.label}</text></g>`
  ).join("");
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#3a4658"/>
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#3a4658"/>
      <text x="10" y="${pad.top + 5}" fill="#aab4c3" font-size="12">${maxY.toFixed(0)} ms</text>
      <text x="18" y="${height - pad.bottom}" fill="#aab4c3" font-size="12">0 ms</text>
      <text x="${pad.left}" y="${height - 16}" fill="#aab4c3" font-size="12">frame ${minX}</text>
      <text x="${width - 105}" y="${height - 16}" fill="#aab4c3" font-size="12">frame ${maxX}</text>
      ${lines}
      ${legend}
    </svg>
  `;
}

function setStatus(message, good = true) {
  statusEl.textContent = message;
  statusEl.style.color = good ? "#8ee59b" : "#ff9b9b";
  if (message) setTimeout(() => { statusEl.textContent = ""; }, 2800);
}

async function loadSessions() {
  const response = await fetch("/api/sessions");
  const data = await response.json();
  sessions = data.sessions || [];
  if (!activeSession && sessions.length) activeSession = sessions[0].id;
  renderSessionList();
  await loadImages();
}

function renderSessionList() {
  sessionList.innerHTML = "";
  if (!sessions.length) {
    sessionList.innerHTML = `<p class="muted">No image sessions found.</p>`;
    return;
  }
  for (const session of sessions) {
    const button = document.createElement("button");
    button.className = "session-item" + (activeSession === session.id ? " active" : "");
    button.textContent = `${session.name} (${session.image_count})`;
    button.title = session.name;
    button.onclick = () => selectSession(session.id);
    sessionList.appendChild(button);
  }
}

async function selectSession(sessionId) {
  activeSession = sessionId;
  activeImage = null;
  boxes = [];
  photo.removeAttribute("src");
  renderAll();
  renderSessionList();
  await loadImages();
}

async function loadImages() {
  const sessionQuery = activeSession === null ? "" : `?session=${encodeURIComponent(activeSession)}`;
  const response = await fetch(`/api/images${sessionQuery}`);
  const data = await response.json();
  images = data.images || [];
  imageCount.textContent = `${images.length} image(s)`;
  renderImageList();
  if (!activeImage && images.length) selectImage(images[0].id);
  if (!images.length) {
    photo.removeAttribute("src");
    imageList.innerHTML = `
      <p class="muted">
        No images found in this session. If this folder contains MP4 videos,
        extract frames first, then launch the UI with <code>--images-dir data/frames</code>.
      </p>
    `;
  }
}

function renderImageList() {
  imageList.innerHTML = "";
  for (const image of images) {
    const button = document.createElement("button");
    button.className = "image-item" + (activeImage?.id === image.id ? " active" : "");
    button.textContent = image.path;
    button.title = image.path;
    button.onclick = () => selectImage(image.id);
    imageList.appendChild(button);
  }
}

async function selectImage(imageId) {
  activeImage = images.find(image => image.id === imageId);
  renderImageList();
  boxes = [];
  photo.src = activeImage.url;
  await new Promise(resolve => { photo.onload = resolve; });
  resizeCanvas();
  const response = await fetch(`/api/annotation?image=${encodeURIComponent(activeImage.id)}`);
  const annotation = await response.json();
  boxes = (annotation.objects || []).map((object, index) => ({
    object_id: object.object_id || `box_${String(index + 1).padStart(3, "0")}`,
    class_name: object.class_name || "painted_number",
    bbox_xyxy: object.bbox_xyxy,
    transcription: object.transcription || "",
    readable: object.readable !== false,
    occluded: object.occluded === true,
  }));
  renderAll();
}

function resizeCanvas() {
  canvas.width = photo.clientWidth;
  canvas.height = photo.clientHeight;
  canvas.style.width = `${photo.clientWidth}px`;
  canvas.style.height = `${photo.clientHeight}px`;
}

function displayToImage(x, y) {
  return [
    Math.round(x * photo.naturalWidth / canvas.width),
    Math.round(y * photo.naturalHeight / canvas.height),
  ];
}

function imageToDisplay(x, y) {
  return [
    x * canvas.width / photo.naturalWidth,
    y * canvas.height / photo.naturalHeight,
  ];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  boxes.forEach((box, index) => {
    const [x1, y1] = imageToDisplay(box.bbox_xyxy[0], box.bbox_xyxy[1]);
    const [x2, y2] = imageToDisplay(box.bbox_xyxy[2], box.bbox_xyxy[3]);
    const isPlate = box.class_name === "license_plate";
    ctx.strokeStyle = isPlate ? "#ffb45e" : "#61dafb";
    ctx.fillStyle = isPlate ? "rgba(255,180,94,0.12)" : "rgba(97,218,251,0.12)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "#101318";
    ctx.fillRect(x1, Math.max(0, y1 - 20), 190, 20);
    ctx.fillStyle = "#eef2f7";
    ctx.fillText(`${index + 1}: ${box.class_name} ${box.transcription || "?"}`, x1 + 4, Math.max(14, y1 - 5));
  });
  if (drawing) {
    ctx.strokeStyle = "#ffcc66";
    ctx.strokeRect(drawing.x, drawing.y, drawing.w, drawing.h);
  }
}

function renderBoxes() {
  boxList.innerHTML = "";
  boxes.forEach((box, index) => {
    const row = document.createElement("div");
    row.className = "box-row";
    row.innerHTML = `
      <strong>Box ${index + 1}</strong>
      <label>Detection class</label>
      <select class="class-name">
        <option value="painted_number" ${box.class_name === "painted_number" ? "selected" : ""}>painted_number</option>
        <option value="license_plate" ${box.class_name === "license_plate" ? "selected" : ""}>license_plate</option>
      </select>
      <label>Text inside the box</label>
      <input type="text" value="${box.transcription}" />
      <label><input type="checkbox" class="readable" ${box.readable ? "checked" : ""}/> readable</label>
      <label><input type="checkbox" class="occluded" ${box.occluded ? "checked" : ""}/> occluded</label>
      <p class="muted">bbox: [${box.bbox_xyxy.join(", ")}]</p>
    `;
    row.querySelector("input[type=text]").oninput = event => {
      const raw = event.target.value.toUpperCase();
      box.transcription = box.class_name === "license_plate"
        ? raw.replace(/[^A-Z0-9]/g, "")
        : raw.replace(/\D/g, "");
      event.target.value = box.transcription;
      draw();
    };
    row.querySelector(".class-name").onchange = event => {
      box.class_name = event.target.value;
      box.transcription = box.class_name === "license_plate"
        ? box.transcription.toUpperCase().replace(/[^A-Z0-9]/g, "")
        : box.transcription.replace(/\D/g, "");
      renderAll();
    };
    row.querySelector(".readable").onchange = event => { box.readable = event.target.checked; };
    row.querySelector(".occluded").onchange = event => { box.occluded = event.target.checked; };
    boxList.appendChild(row);
  });
}

function renderAll() {
  renderBoxes();
  draw();
}

canvas.addEventListener("mousedown", event => {
  const rect = canvas.getBoundingClientRect();
  drawing = { x: event.clientX - rect.left, y: event.clientY - rect.top, w: 0, h: 0 };
});

canvas.addEventListener("mousemove", event => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  drawing.w = event.clientX - rect.left - drawing.x;
  drawing.h = event.clientY - rect.top - drawing.y;
  draw();
});

canvas.addEventListener("mouseup", () => {
  if (!drawing) return;
  const x1d = Math.min(drawing.x, drawing.x + drawing.w);
  const y1d = Math.min(drawing.y, drawing.y + drawing.h);
  const x2d = Math.max(drawing.x, drawing.x + drawing.w);
  const y2d = Math.max(drawing.y, drawing.y + drawing.h);
  drawing = null;
  if (Math.abs(x2d - x1d) < 5 || Math.abs(y2d - y1d) < 5) {
    draw();
    return;
  }
  const [x1, y1] = displayToImage(x1d, y1d);
  const [x2, y2] = displayToImage(x2d, y2d);
  boxes.push({
    object_id: `box_${String(boxes.length + 1).padStart(3, "0")}`,
    class_name: newBoxClass.value,
    bbox_xyxy: [x1, y1, x2, y2],
    transcription: "",
    readable: true,
    occluded: false,
  });
  renderAll();
});

document.getElementById("undo").onclick = () => { boxes.pop(); renderAll(); };
document.getElementById("clear").onclick = () => { boxes = []; renderAll(); };
document.getElementById("refresh").onclick = loadSessions;
document.getElementById("refreshPredictions").onclick = loadPredictions;
document.getElementById("refreshBenchmarks").onclick = () => loadBenchmarks(true);
document.getElementById("refreshFieldCaptures").onclick = loadFieldCaptures;
toggleBenchmarkPolling.onclick = () => {
  benchmarkPollingEnabled = !benchmarkPollingEnabled;
  toggleBenchmarkPolling.textContent = benchmarkPollingEnabled ? "Pause live updates" : "Resume live updates";
  startBenchmarkPolling();
};
document.querySelectorAll(".view-tab").forEach(tab => {
  tab.onclick = () => selectView(tab.dataset.view);
});
document.getElementById("save").onclick = async () => {
  if (!activeImage) return;
  const payload = {
    image_id: activeImage.id,
    width: photo.naturalWidth,
    height: photo.naturalHeight,
    objects: boxes,
  };
  const response = await fetch("/api/annotation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (data.ok) setStatus("Saved");
  else setStatus(data.error || "Save failed", false);
};

window.addEventListener("resize", () => {
  if (photo.src) {
    resizeCanvas();
    draw();
  }
});

loadConfig().catch(error => setStatus(error.message, false));
loadSessions().catch(error => setStatus(error.message, false));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
