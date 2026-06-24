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
    ImageAnnotation,
    PAINTED_NUMBER_CLASS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class WorkbenchConfig:
    images_dir: Path
    annotations_dir: Path
    session_id: str | None
    host: str
    port: int


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
                class_name=PAINTED_NUMBER_CLASS,
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
                    }
                )
            elif parsed.path == "/api/sessions":
                self.send_json({"sessions": list_sessions(self.config.images_dir)})
            elif parsed.path == "/api/images":
                query = parse_qs(parsed.query)
                session_id = query.get("session", [self.config.session_id or ""])[0] or None
                self.send_json({"images": list_images(self.config.images_dir, session_id=session_id)})
            elif parsed.path == "/api/annotation":
                query = parse_qs(parsed.query)
                image_id = query.get("image", [""])[0]
                self.handle_get_annotation(image_id)
            elif parsed.path.startswith("/images/"):
                self.handle_get_image(parsed.path.removeprefix("/images/"))
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
    header { padding: 14px 20px; border-bottom: 1px solid #293241; display: flex; gap: 16px; align-items: center; }
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
  </style>
</head>
<body>
  <header>
    <h1>Container ID Workbench</h1>
    <span class="muted">Draw boxes around painted numbers, type the digits, save JSON.</span>
    <span id="status"></span>
  </header>
  <main>
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
      <p class="muted">Tip: drag on the image to create a box. Use one box per visible number.</p>
      <div id="boxList"></div>
    </aside>
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

let images = [];
let sessions = [];
let activeSession = null;
let activeImage = null;
let boxes = [];
let drawing = null;

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
    ctx.strokeStyle = "#61dafb";
    ctx.fillStyle = "rgba(97,218,251,0.12)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "#101318";
    ctx.fillRect(x1, Math.max(0, y1 - 20), 72, 20);
    ctx.fillStyle = "#eef2f7";
    ctx.fillText(`${index + 1}: ${box.transcription || "?"}`, x1 + 4, Math.max(14, y1 - 5));
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
      <label>Digits inside the box</label>
      <input type="text" inputmode="numeric" pattern="[0-9]*" value="${box.transcription}" />
      <label><input type="checkbox" class="readable" ${box.readable ? "checked" : ""}/> readable</label>
      <label><input type="checkbox" class="occluded" ${box.occluded ? "checked" : ""}/> occluded</label>
      <p class="muted">bbox: [${box.bbox_xyxy.join(", ")}]</p>
    `;
    row.querySelector("input[type=text]").oninput = event => {
      box.transcription = event.target.value.replace(/\D/g, "");
      event.target.value = box.transcription;
      draw();
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
