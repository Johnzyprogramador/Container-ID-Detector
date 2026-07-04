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
    .benchmark-sidebar { border-right: 1px solid #293241; padding: 12px; overflow: auto; }
    .benchmark-content { padding: 18px; overflow: auto; }
    .benchmark-item { width: 100%; text-align: left; margin-bottom: 6px; }
    .benchmark-item.active { outline: 2px solid #8ee59b; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; margin: 12px 0; }
    .metric-card { border: 1px solid #293241; border-radius: 10px; background: #151a22; padding: 12px; }
    .metric-card strong { display: block; font-size: 20px; margin-top: 4px; }
    .run-card { border: 1px solid #293241; border-radius: 10px; padding: 12px; margin: 12px 0; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .plot-grid img { width: 100%; background: white; border-radius: 8px; }
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
      <div class="toolbar"><button id="refreshBenchmarks">Refresh benchmarks</button></div>
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
  if (benchmarkPoll) clearInterval(benchmarkPoll);
  if (viewId === "benchmarkView") {
    loadBenchmarks().catch(error => setStatus(error.message, false));
    benchmarkPoll = setInterval(() => loadBenchmarks(true).catch(() => {}), 5000);
  }
}

async function loadBenchmarks(preserveSelection = false) {
  const response = await fetch("/api/benchmarks");
  const data = await response.json();
  benchmarkMatrices = data.matrices || [];
  if ((!preserveSelection || !activeBenchmark) && benchmarkMatrices.length) activeBenchmark = benchmarkMatrices[0].id;
  renderBenchmarkList();
  renderBenchmarkDetails();
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
        ${metricCard("CPU temperature", `${formatNumber(resource.cpu_temperature_c)} °C`)}
        ${metricCard("CPU power", `${formatNumber(resource.cpu_power_w)} W`)}
        ${metricCard("GPU", `${formatNumber(resource.gpu_util_percent)}%`)}
        ${metricCard("GPU memory", `${formatNumber(resource.gpu_memory_mb)} MB`)}
        ${metricCard("GPU temperature", `${formatNumber(resource.gpu_temperature_c)} °C`)}
        ${metricCard("GPU power", `${formatNumber(resource.gpu_power_w)} W`)}
      </div>
      <div class="plot-grid">${run.plots.map(url => `<img src="${url}?t=${Date.now()}" alt="Benchmark plot">`).join("")}</div>
    `;
    benchmarkRuns.appendChild(card);
  });
  comparisonPlots.innerHTML = selected.comparison_plots
    .map(url => `<img src="${url}?t=${Date.now()}" alt="Comparison plot">`).join("");
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
