#!/usr/bin/env python3
"""Train an Ultralytics YOLO detector from an exported dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-class YOLO detector.")
    parser.add_argument("--dataset", required=True, help="Path to exported data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or weights path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu")
    parser.add_argument("--name", default="temporary_two_class_detector")
    parser.add_argument(
        "--project",
        default=str(REPO_ROOT / "runs" / "detection"),
        help="Training output directory (defaults inside this repository).",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[detection]'") from exc

    dataset = Path(args.dataset).resolve()
    if not dataset.is_file():
        raise SystemExit(f"Dataset configuration not found: {dataset}")
    model = YOLO(args.model)
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
