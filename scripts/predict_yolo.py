#!/usr/bin/env python3
"""Run a trained YOLO detector over images or videos and save visual results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".mov", ".avi", ".mkv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def find_media(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in MEDIA_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
        )
    return []


def output_path_for(source_root: Path, media_path: Path, output_root: Path) -> Path:
    relative = Path(media_path.name) if source_root == media_path else media_path.relative_to(source_root)
    if media_path.suffix.lower() in VIDEO_EXTENSIONS:
        relative = relative.with_name(f"{relative.stem}_predicted.mp4")
    else:
        relative = relative.with_name(f"{relative.stem}_predicted{relative.suffix}")
    return output_root / relative


def process_media(model, media_path: Path, destination: Path, args) -> dict:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    is_video = media_path.suffix.lower() in VIDEO_EXTENSIONS
    writer = None
    frames = 0
    detections: Counter[str] = Counter()

    if is_video:
        capture = cv2.VideoCapture(str(media_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
    else:
        fps = None

    results = model.predict(
        source=str(media_path),
        stream=True,
        conf=args.confidence,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
    )
    for result in results:
        rendered = result.plot()
        if is_video and writer is None:
            height, width = rendered.shape[:2]
            writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {destination}")
        if writer is not None:
            writer.write(rendered)
        else:
            if not cv2.imwrite(str(destination), rendered):
                raise RuntimeError(f"Could not save output image: {destination}")

        frames += 1
        if result.boxes is not None:
            for class_id in result.boxes.cls.tolist():
                detections[result.names[int(class_id)]] += 1

    if writer is not None:
        writer.release()
    return {
        "source": str(media_path),
        "output": str(destination),
        "frames": frames,
        "detections": dict(sorted(detections.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions on images or videos.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="One media file or a directory searched recursively")
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "predictions"))
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install detection dependencies with: pip install -e '.[detection]'") from exc

    source = Path(args.source).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    media = find_media(source)
    if not media:
        raise SystemExit(f"No supported images or videos found under: {source}")

    model = YOLO(str(weights))
    summaries = []
    for index, media_path in enumerate(media, start=1):
        destination = output_path_for(source, media_path, output)
        print(f"[{index}/{len(media)}] {media_path} -> {destination}")
        summaries.append(process_media(model, media_path, destination, args))

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({"files": summaries}, indent=2) + "\n")
    print(f"Prediction summary: {summary_path}")


if __name__ == "__main__":
    main()
