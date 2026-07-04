#!/usr/bin/env python3
"""Run an unattended four-camera CPU/GPU benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from container_vision.benchmarking import (  # noqa: E402
    generate_comparison_plots,
    generate_run_plots,
    summarize_rows,
)
from container_vision.recognition import EasyTextRecognizer  # noqa: E402
from scripts.predict_yolo import find_media  # noqa: E402
from scripts.predict_yolo_ocr import crop_detections  # noqa: E402

FRAME_FIELDS = [
    "timestamp", "elapsed_s", "mode", "fps_per_camera", "camera_id", "frame_index",
    "warmup", "decode_ms", "queue_wait_ms", "queue_depth", "yolo_batch_size",
    "ocr_batch_size", "yolo_preprocess_ms", "yolo_inference_ms",
    "yolo_postprocess_ms", "crop_extraction_ms", "ocr_preprocess_ms", "ocr_inference_ms",
    "ocr_postprocess_ms", "rules_event_ms", "frame_processing_ms", "end_to_end_ms",
    "detections", "ocr_crops", "decode_failed",
]
RESOURCE_FIELDS = [
    "timestamp", "elapsed_s", "cpu_percent", "process_cpu_percent", "ram_percent",
    "process_ram_mb", "cpu_temperature_c", "gpu_util_percent", "gpu_memory_mb",
    "gpu_memory_percent", "gpu_temperature_c", "gpu_power_w", "cpu_power_w",
]
EVENT_FIELDS = [
    "timestamp", "elapsed_s", "camera_id", "class_name", "decision", "observations",
    "event_latency_ms", "decision_compute_ms",
]
TIMING_FIELDS = [
    "decode_ms", "queue_wait_ms", "yolo_preprocess_ms", "yolo_inference_ms",
    "yolo_postprocess_ms", "crop_extraction_ms", "ocr_preprocess_ms", "ocr_inference_ms",
    "ocr_postprocess_ms", "rules_event_ms", "frame_processing_ms", "end_to_end_ms",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def preflight(args) -> dict:
    try:
        import cv2
        import easyocr
        import matplotlib
        import psutil
        import torch
        import ultralytics
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependencies. Run: pip install -e "
            "'.[detection,recognition,benchmark]'"
        ) from exc
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    if "gpu" in args.modes and not torch.cuda.is_available():
        raise SystemExit("GPU mode requested but PyTorch reports CUDA unavailable")
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "ram_gb": memory.total / 1024**3,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "easyocr": getattr(easyocr, "__version__", "unknown"),
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
    }


def assign_camera_playlists(videos: list[Path], cameras: int = 4) -> list[list[Path]]:
    if len(videos) < cameras:
        raise ValueError(f"At least {cameras} videos are required; found {len(videos)}")
    playlists = [[] for _ in range(cameras)]
    for index, video in enumerate(sorted(videos)):
        playlists[index % cameras].append(video)
    return playlists


class CameraPlaylist:
    def __init__(self, paths: list[Path]) -> None:
        import cv2

        self.cv2 = cv2
        self.paths = paths
        self.path_index = 0
        self.capture = None
        self.frame_index = 0
        self._open_current()

    def _open_current(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = self.cv2.VideoCapture(str(self.paths[self.path_index]))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.paths[self.path_index]}")

    def read(self):
        started = time.perf_counter_ns()
        ok, frame = self.capture.read()
        if not ok:
            self.path_index = (self.path_index + 1) % len(self.paths)
            self._open_current()
            ok, frame = self.capture.read()
        finished = time.perf_counter_ns()
        if not ok:
            return None, (finished - started) / 1_000_000, self.frame_index
        index = self.frame_index
        self.frame_index += 1
        return frame, (finished - started) / 1_000_000, index

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


@dataclass
class VoteWindow:
    started_at: float
    observations: list[tuple[str, float]]


class EventDecider:
    def __init__(self, observations: int = 5, timeout_s: float = 2.0) -> None:
        self.required = observations
        self.timeout_s = timeout_s
        self.windows: dict[tuple[int, str], VoteWindow] = {}

    def add(self, camera_id: int, class_name: str, text: str, confidence: float, now: float) -> dict | None:
        key = (camera_id, class_name)
        window = self.windows.setdefault(key, VoteWindow(now, []))
        window.observations.append((text or "UNKNOWN", confidence))
        if len(window.observations) < self.required and now - window.started_at < self.timeout_s:
            return None
        scores: dict[str, float] = defaultdict(float)
        for value, weight in window.observations:
            scores[value] += max(weight, 0.01)
        decision = max(scores, key=scores.get)
        result = {
            "camera_id": camera_id,
            "class_name": class_name,
            "decision": decision,
            "observations": len(window.observations),
            "event_latency_ms": (now - window.started_at) * 1000,
        }
        self.windows.pop(key, None)
        return result


class ResourceSampler(threading.Thread):
    def __init__(self, output: Path, started_at: float, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.output = output
        self.started_at = started_at
        self.stop_event = stop_event
        self.rows: list[dict] = []
        self.last_cpu_energy: tuple[float, float] | None = None

    def run(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        process.cpu_percent(None)
        gpu = self._open_gpu()
        with self.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESOURCE_FIELDS)
            writer.writeheader()
            while not self.stop_event.is_set():
                row = self._sample(psutil, process, gpu)
                self.rows.append(row)
                writer.writerow(row)
                stream.flush()
                self.stop_event.wait(1.0)
        if gpu is not None:
            try:
                gpu[0].nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _open_gpu():
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

    def _sample(self, psutil, process, gpu) -> dict:
        cpu_temp = ""
        try:
            temperatures = psutil.sensors_temperatures()
            values = [entry.current for entries in temperatures.values() for entry in entries if entry.current]
            cpu_temp = max(values) if values else ""
        except Exception:
            pass
        cpu_power = self._cpu_power()
        gpu_values = {key: "" for key in (
            "gpu_util_percent", "gpu_memory_mb", "gpu_memory_percent", "gpu_temperature_c", "gpu_power_w"
        )}
        if gpu is not None:
            try:
                pynvml, handle = gpu
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_values = {
                    "gpu_util_percent": utilization.gpu,
                    "gpu_memory_mb": memory.used / 1024 / 1024,
                    "gpu_memory_percent": memory.used / memory.total * 100,
                    "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                }
            except Exception:
                pass
        return {
            "timestamp": utc_now(),
            "elapsed_s": time.monotonic() - self.started_at,
            "cpu_percent": psutil.cpu_percent(None),
            "process_cpu_percent": process.cpu_percent(None),
            "ram_percent": psutil.virtual_memory().percent,
            "process_ram_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_temperature_c": cpu_temp,
            "cpu_power_w": cpu_power,
            **gpu_values,
        }

    def _cpu_power(self):
        try:
            energy_paths = list(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"))
            if not energy_paths:
                return ""
            energy_j = sum(float(path.read_text().strip()) for path in energy_paths) / 1_000_000
            now = time.monotonic()
            previous = self.last_cpu_energy
            self.last_cpu_energy = (energy_j, now)
            if previous is None or energy_j < previous[0]:
                return ""
            return (energy_j - previous[0]) / max(now - previous[1], 0.001)
        except Exception:
            return ""


def yolo_predict(model, frames: list, mode: str, args):
    options = dict(conf=args.confidence, imgsz=args.image_size, device="cpu" if mode == "cpu" else 0, verbose=False)
    if mode == "gpu":
        return list(model.predict(source=frames, **options))
    return [model.predict(source=frame, **options)[0] for frame in frames]


def run_one(args, run_dir: Path, mode: str, fps: int, videos: list[Path]) -> dict:
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": mode, "fps_per_camera": fps, "cameras": 4,
        "duration_s": args.duration, "warmup_s": args.warmup,
        "weights": str(Path(args.weights).resolve()), "source": str(Path(args.source).resolve()),
        "image_size": args.image_size, "confidence": args.confidence,
        "ocr_batch": args.ocr_batch, "started_at": utc_now(),
    }
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "status.json", {**config, "status": "initializing"})
    playlists = assign_camera_playlists(videos)
    cameras = [CameraPlaylist(paths) for paths in playlists]
    model = YOLO(config["weights"])
    recognizer = EasyTextRecognizer(device="cpu" if mode == "cpu" else "gpu:0", batch_size=args.ocr_batch)
    decider = EventDecider()
    stop_resources = threading.Event()
    started_at = time.monotonic()
    sampler = ResourceSampler(run_dir / "resource_metrics.csv", started_at, stop_resources)
    sampler.start()
    period = 1.0 / fps
    scheduled = started_at
    processed = 0
    dropped = 0
    decode_failures = 0
    frame_rows: list[dict] = []
    event_rows: list[dict] = []
    last_live_write = 0.0

    try:
        with (run_dir / "frame_metrics.csv").open("w", newline="") as frame_stream, \
             (run_dir / "event_metrics.csv").open("w", newline="") as event_stream:
            frame_writer = csv.DictWriter(frame_stream, fieldnames=FRAME_FIELDS)
            event_writer = csv.DictWriter(event_stream, fieldnames=EVENT_FIELDS)
            frame_writer.writeheader()
            event_writer.writeheader()
            atomic_json(run_dir / "status.json", {**config, "status": "running"})

            while True:
                now = time.monotonic()
                elapsed = now - started_at
                if elapsed >= args.duration:
                    break
                if now < scheduled:
                    time.sleep(scheduled - now)
                tick_started = time.monotonic()
                lag = max(0.0, tick_started - scheduled)
                queue_depth = int(lag // period)
                if lag >= period:
                    missed = queue_depth
                    dropped += missed * len(cameras)
                    scheduled += missed * period
                    lag = max(0.0, tick_started - scheduled)

                frames, decode_times, frame_indices, failed = [], [], [], []
                for camera in cameras:
                    frame, decode_ms, frame_index = camera.read()
                    frames.append(frame)
                    decode_times.append(decode_ms)
                    frame_indices.append(frame_index)
                    failed.append(frame is None)
                decode_failures += sum(failed)
                valid = [(index, frame) for index, frame in enumerate(frames) if frame is not None]
                if not valid:
                    scheduled += period
                    continue

                results = yolo_predict(model, [frame for _, frame in valid], mode, args)
                per_camera_detections: dict[int, list[dict]] = defaultdict(list)
                crop_times: dict[int, float] = defaultdict(float)
                all_crops, all_metadata = [], []
                for (camera_index, _), result in zip(valid, results):
                    crop_started = time.perf_counter_ns()
                    crops, metadata = crop_detections(result)
                    crop_times[camera_index] = (time.perf_counter_ns() - crop_started) / 1_000_000
                    for item in metadata:
                        item["camera_id"] = camera_index
                    all_crops.extend(crops)
                    all_metadata.extend(metadata)
                    per_camera_detections[camera_index].extend(metadata)

                ocr_results, ocr_times = recognizer.recognize(
                    all_crops, [item["class_name"] for item in all_metadata]
                )
                rules_started = time.perf_counter_ns()
                for metadata, ocr_result in zip(all_metadata, ocr_results):
                    metadata.update(text=ocr_result.text, ocr_confidence=ocr_result.confidence)
                    event_started = time.perf_counter_ns()
                    event = decider.add(
                        metadata["camera_id"], metadata["class_name"], ocr_result.text,
                        ocr_result.confidence, time.monotonic(),
                    )
                    if event:
                        event_row = {
                            "timestamp": utc_now(), "elapsed_s": time.monotonic() - started_at,
                            **event, "decision_compute_ms": (time.perf_counter_ns() - event_started) / 1_000_000,
                        }
                        event_rows.append(event_row)
                        event_writer.writerow(event_row)
                        event_stream.flush()
                rules_ms = (time.perf_counter_ns() - rules_started) / 1_000_000
                tick_finished = time.monotonic()
                warmup = int(tick_finished - started_at < args.warmup)

                for result_index, (camera_index, _) in enumerate(valid):
                    result = results[result_index]
                    row = {
                        "timestamp": utc_now(), "elapsed_s": tick_finished - started_at,
                        "mode": mode, "fps_per_camera": fps, "camera_id": camera_index,
                        "frame_index": frame_indices[camera_index], "warmup": warmup,
                        "decode_ms": decode_times[camera_index], "queue_wait_ms": lag * 1000,
                        "queue_depth": queue_depth, "yolo_batch_size": len(valid),
                        "ocr_batch_size": len(all_crops),
                        "yolo_preprocess_ms": float(result.speed.get("preprocess", 0.0)),
                        "yolo_inference_ms": float(result.speed.get("inference", 0.0)),
                        "yolo_postprocess_ms": float(result.speed.get("postprocess", 0.0)),
                        "crop_extraction_ms": crop_times[camera_index],
                        **ocr_times, "rules_event_ms": rules_ms,
                        "frame_processing_ms": (tick_finished - tick_started) * 1000,
                        "end_to_end_ms": (tick_finished - scheduled) * 1000,
                        "detections": len(per_camera_detections[camera_index]),
                        "ocr_crops": len(per_camera_detections[camera_index]), "decode_failed": 0,
                    }
                    frame_rows.append(row)
                    frame_writer.writerow(row)
                    processed += 1
                frame_stream.flush()
                scheduled += period

                elapsed = tick_finished - started_at
                if elapsed - last_live_write >= 1.0:
                    measured = [row for row in frame_rows if not row["warmup"]]
                    latency = summarize_rows(measured[-1000:], ["end_to_end_ms"])["end_to_end_ms"]
                    atomic_json(run_dir / "live.json", {
                        **config, "status": "running", "elapsed_s": elapsed,
                        "progress_percent": min(100, elapsed / args.duration * 100),
                        "processed_frames": processed, "dropped_frames": dropped,
                        "queue_depth": queue_depth,
                        "processed_fps": processed / max(elapsed, 0.001),
                        "latency_ms": latency,
                        "latest_resource": sampler.rows[-1] if sampler.rows else {},
                    })
                    last_live_write = elapsed
    finally:
        stop_resources.set()
        sampler.join(timeout=5)
        for camera in cameras:
            camera.close()

    measured_rows = [row for row in frame_rows if not row["warmup"]]
    measured_duration = max(args.duration - args.warmup, 0.001)
    expected = int(args.duration * fps * len(cameras))
    summary = {
        **config, "status": "complete", "finished_at": utc_now(),
        "processed_frames": processed, "expected_frames": expected,
        "dropped_frames": dropped, "decode_failures": decode_failures,
        "dropped_percent": dropped / max(expected, 1) * 100,
        "processed_fps": len(measured_rows) / measured_duration,
        "latencies_ms": summarize_rows(measured_rows, TIMING_FIELDS),
        "resources": summarize_rows(sampler.rows, [
            "cpu_percent", "process_cpu_percent", "ram_percent", "process_ram_mb",
            "gpu_util_percent", "gpu_memory_mb", "gpu_memory_percent",
            "gpu_temperature_c", "gpu_power_w",
            "cpu_power_w",
        ]),
        "events": len(event_rows),
    }
    atomic_json(run_dir / "summary.json", summary)
    try:
        plots = generate_run_plots(run_dir)
    except Exception as exc:
        plots = []
        summary["plot_error"] = str(exc)
        (run_dir / "plot_error.log").write_text(traceback.format_exc())
    summary["plots"] = plots
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "live.json", summary)
    atomic_json(run_dir / "status.json", {**config, "status": "complete", "summary": "summary.json"})
    del recognizer, model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-camera CPU/GPU benchmark matrix.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--output-root", default="runs/benchmarks")
    parser.add_argument("--modes", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    parser.add_argument("--fps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--warmup", type=float, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--matrix-id", default=None, help="Reuse this id to resume an interrupted matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup >= args.duration:
        raise SystemExit("--warmup must be shorter than --duration")
    hardware = preflight(args)
    videos = [path for path in find_media(Path(args.source).resolve()) if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if len(videos) < 4:
        raise SystemExit(f"Need at least four videos; found {len(videos)}")
    matrix_id = args.matrix_id or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    matrix_dir = Path(args.output_root).resolve() / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(matrix_dir / "hardware.json", hardware)
    matrix = {
        "matrix_id": matrix_id, "status": "running", "started_at": utc_now(),
        "modes": args.modes, "fps": args.fps, "duration_s": args.duration,
        "warmup_s": args.warmup, "runs": [],
    }
    atomic_json(matrix_dir / "matrix.json", matrix)

    for mode in args.modes:
        for fps in args.fps:
            run_name = f"{mode}_{fps}fps"
            run_dir = matrix_dir / run_name
            status_path = run_dir / "status.json"
            if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "complete":
                matrix["runs"].append({"name": run_name, "status": "complete", "resumed": True})
                continue
            try:
                print(f"\n=== Starting {run_name} ===", flush=True)
                run_one(args, run_dir, mode, fps, videos)
                matrix["runs"].append({"name": run_name, "status": "complete"})
            except Exception as exc:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "error.log").write_text(traceback.format_exc())
                atomic_json(status_path, {"status": "failed", "error": str(exc)})
                matrix["runs"].append({"name": run_name, "status": "failed", "error": str(exc)})
                print(f"FAILED {run_name}: {exc}", file=sys.stderr, flush=True)
            atomic_json(matrix_dir / "matrix.json", matrix)

    try:
        matrix["comparison_plots"] = generate_comparison_plots(matrix_dir)
    except Exception as exc:
        matrix["plot_error"] = str(exc)
        (matrix_dir / "plot_error.log").write_text(traceback.format_exc())
    matrix["status"] = "complete"
    matrix["finished_at"] = utc_now()
    atomic_json(matrix_dir / "matrix.json", matrix)
    print(f"\nBenchmark matrix complete: {matrix_dir}", flush=True)


if __name__ == "__main__":
    main()
