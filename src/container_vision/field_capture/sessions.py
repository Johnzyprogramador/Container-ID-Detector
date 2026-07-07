from __future__ import annotations

import csv
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
METRIC_FIELDS = (
    "sequence",
    "capture_time_ms",
    "server_receive_time_ms",
    "decode_ms",
    "inference_ms",
    "server_total_ms",
    "logical_streams",
    "detections",
    "jpeg_bytes",
    "client_skipped",
)
CLIENT_METRIC_FIELDS = ("sequence", "client_receive_time_ms", "round_trip_ms")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_id(value: str, label: str = "identifier") -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {label}")
    return value


@dataclass
class CaptureSession:
    session_id: str
    root: Path
    logical_streams: int
    device_name: str = "iPhone"
    status: str = "recording"
    started_at: str = field(default_factory=utc_now)
    stopped_at: str | None = None
    frames_received: int = 0
    detections: int = 0
    segments_received: int = 0
    bytes_received: int = 0
    latest_result: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def directory(self) -> Path:
        return self.root / self.session_id

    def initialize(self) -> None:
        for name in ("cloud_recording", "inference_frames", "results", "metrics"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.save_metadata()

    def metadata(self) -> dict:
        return {
            "session_id": self.session_id,
            "device_name": self.device_name,
            "logical_streams": self.logical_streams,
            "status": self.status,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frames_received": self.frames_received,
            "detections": self.detections,
            "segments_received": self.segments_received,
            "bytes_received": self.bytes_received,
            "latest_result": self.latest_result,
        }

    def save_metadata(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "session.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.metadata(), indent=2) + "\n")
        temporary.replace(target)

    def save_segment(self, index: int, content: bytes, extension: str) -> Path:
        extension = extension.lower().lstrip(".")
        if extension not in {"mp4", "webm"}:
            extension = "webm"
        target = self.directory / "cloud_recording" / f"segment_{index:05d}.{extension}"
        target.write_bytes(content)
        with self._lock:
            self.segments_received += 1
            self.bytes_received += len(content)
            self.save_metadata()
        return target

    def save_frame_result(self, sequence: int, jpeg: bytes, result: dict) -> None:
        frame_name = f"frame_{sequence:08d}"
        (self.directory / "inference_frames" / f"{frame_name}.jpg").write_bytes(jpeg)
        (self.directory / "results" / f"{frame_name}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        metrics_path = self.directory / "metrics" / "frames.csv"
        write_header = not metrics_path.exists()
        metrics = result["metrics"]
        row = {name: metrics.get(name, "") for name in METRIC_FIELDS}
        with metrics_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with self._lock:
            self.frames_received += 1
            self.detections += len(result.get("detections", []))
            self.bytes_received += len(jpeg)
            self.latest_result = result
            self.save_metadata()

    def save_client_metric(self, payload: dict) -> None:
        metrics_path = self.directory / "metrics" / "client.csv"
        write_header = not metrics_path.exists()
        row = {name: payload.get(name, "") for name in CLIENT_METRIC_FIELDS}
        with metrics_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CLIENT_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with self._lock:
            if self.latest_result.get("sequence") == payload.get("sequence"):
                self.latest_result.setdefault("metrics", {})["client_round_trip_ms"] = payload.get(
                    "round_trip_ms"
                )
                self.save_metadata()

    def stop(self) -> None:
        with self._lock:
            self.status = "complete"
            self.stopped_at = utc_now()
            self.save_metadata()


class SessionStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, CaptureSession] = {}
        self._lock = threading.Lock()

    def create(self, *, logical_streams: int, device_name: str = "iPhone") -> CaptureSession:
        if logical_streams not in {1, 2}:
            raise ValueError("logical_streams must be 1 or 2")
        session_id = datetime.now(timezone.utc).strftime("capture_%Y%m%d_%H%M%S_%f")
        session = CaptureSession(session_id, self.root, logical_streams, device_name=device_name)
        session.initialize()
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> CaptureSession:
        validate_id(session_id, "session id")
        with self._lock:
            existing = self._sessions.get(session_id)
        if existing:
            return existing
        metadata_path = self.root / session_id / "session.json"
        if not metadata_path.is_file():
            raise KeyError(session_id)
        data = json.loads(metadata_path.read_text())
        session = CaptureSession(
            session_id=session_id,
            root=self.root,
            logical_streams=int(data["logical_streams"]),
            device_name=data.get("device_name", "iPhone"),
            status=data.get("status", "complete"),
            started_at=data.get("started_at", utc_now()),
            stopped_at=data.get("stopped_at"),
            frames_received=int(data.get("frames_received", 0)),
            detections=int(data.get("detections", 0)),
            segments_received=int(data.get("segments_received", 0)),
            bytes_received=int(data.get("bytes_received", 0)),
            latest_result=data.get("latest_result", {}),
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def list(self) -> list[dict]:
        sessions = []
        for path in sorted(self.root.glob("*/session.json"), reverse=True):
            try:
                sessions.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sessions
