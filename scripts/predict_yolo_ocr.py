#!/usr/bin/env python3
"""Run YOLO and batched PaddleOCR over local videos with stage timings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.recognition import PaddleTextRecognizer  # noqa: E402
from scripts.predict_yolo import (  # noqa: E402
    VIDEO_EXTENSIONS,
    build_browser_transcode_command,
    find_media,
    output_path_for,
)


def crop_detections(result) -> tuple[list, list[dict]]:
    frame = result.orig_img
    height, width = frame.shape[:2]
    crops, metadata = [], []
    if result.boxes is None:
        return crops, metadata
    for xyxy, class_id, confidence in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        class_name = result.names[int(class_id)]
        crops.append(frame[y1:y2, x1:x2].copy())
        metadata.append(
            {
                "class_name": class_name,
                "detector_confidence": float(confidence),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return crops, metadata


def draw_detections(frame, detections: list[dict]) -> None:
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        color = (255, 200, 70) if detection["class_name"] == "painted_number" else (70, 170, 255)
        label = (
            f"{detection['class_name']} {detection['detector_confidence']:.2f} "
            f"OCR:{detection['text'] or '?'} {detection['ocr_confidence']:.2f}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(text_height + 8, y1)
        cv2.rectangle(frame, (x1, label_y - text_height - 8), (x1 + text_width + 6, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)


def process_video(model, recognizer, source: Path, destination: Path, args) -> dict:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    working = destination.with_name(f".{destination.stem}.working.mp4")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()
    writer = None
    frame_metrics = []
    prediction_iter = model.predict(
        source=str(source), stream=True, conf=args.confidence,
        imgsz=args.image_size, device=args.yolo_device, verbose=False,
    )

    for frame_index, result in enumerate(prediction_iter):
        frame_started = perf_counter_ns()
        crop_started = perf_counter_ns()
        crops, detections = crop_detections(result)
        crop_finished = perf_counter_ns()
        ocr_results, ocr_times = recognizer.recognize(
            crops, [item["class_name"] for item in detections]
        )
        for detection, ocr_result in zip(detections, ocr_results):
            detection.update(
                raw_text=ocr_result.raw_text,
                text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
            )
        rendered = result.orig_img.copy()
        draw_detections(rendered, detections)
        if writer is None:
            height, width = rendered.shape[:2]
            writer = cv2.VideoWriter(
                str(working), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {working}")
        writer.write(rendered)
        frame_finished = perf_counter_ns()
        frame_metrics.append(
            {
                "frame_index": frame_index,
                "detections": detections,
                "timings_ms": {
                    "yolo_preprocess": float(result.speed.get("preprocess", 0.0)),
                    "yolo_inference": float(result.speed.get("inference", 0.0)),
                    "yolo_postprocess": float(result.speed.get("postprocess", 0.0)),
                    "crop_extraction": (crop_finished - crop_started) / 1_000_000,
                    **ocr_times,
                    "post_ocr_frame_work": (frame_finished - crop_finished) / 1_000_000
                    - sum(ocr_times.values()),
                },
            }
        )

    if writer is not None:
        writer.release()
        try:
            subprocess.run(build_browser_transcode_command(working, destination), check=True)
        finally:
            working.unlink(missing_ok=True)
    return {"source": str(source), "output": str(destination), "frames": frame_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO plus PaddleOCR predictions.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "ocr_predictions"))
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--ocr-device", default="gpu:0", help="Paddle device: gpu:0 or cpu")
    parser.add_argument("--ocr-model", default="en_PP-OCRv5_mobile_rec")
    parser.add_argument("--ocr-batch", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install detection dependencies with: pip install -e '.[detection]'") from exc

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    media = [item for item in find_media(source) if item.suffix.lower() in VIDEO_EXTENSIONS]
    if not media:
        raise SystemExit(f"No videos found under: {source}")
    model = YOLO(str(Path(args.weights).resolve()))
    recognizer = PaddleTextRecognizer(
        device=args.ocr_device, model_name=args.ocr_model, batch_size=args.ocr_batch
    )
    summaries = []
    for index, media_path in enumerate(media, start=1):
        destination = output_path_for(source, media_path, output)
        print(f"[{index}/{len(media)}] {media_path} -> {destination}")
        summaries.append(process_video(model, recognizer, media_path, destination, args))
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({"files": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
