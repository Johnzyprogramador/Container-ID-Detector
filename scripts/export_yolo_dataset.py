#!/usr/bin/env python3
"""Export canonical annotations into an immutable Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.data import export_yolo_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a versioned two-class YOLO dataset.")
    parser.add_argument("--frames-dir", default="data/frames")
    parser.add_argument("--annotations-dir", default="data/annotations/manual")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validation-sessions",
        nargs="+",
        required=True,
        help="Complete session IDs reserved for validation, for example session_006.",
    )
    args = parser.parse_args()
    manifest = export_yolo_dataset(
        frames_dir=Path(args.frames_dir),
        annotations_dir=Path(args.annotations_dir),
        output_dir=Path(args.output),
        validation_sessions=set(args.validation_sessions),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
