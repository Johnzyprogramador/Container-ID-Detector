#!/usr/bin/env python3
"""Audit local frames and canonical annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.data import audit_annotations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit annotation coverage and validity.")
    parser.add_argument("--frames-dir", default="data/frames")
    parser.add_argument("--annotations-dir", default="data/annotations/manual")
    args = parser.parse_args()
    result = audit_annotations(Path(args.frames_dir), Path(args.annotations_dir))
    print(json.dumps(result.to_dict(), indent=2))
    if result.invalid_annotations or result.missing_images:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
