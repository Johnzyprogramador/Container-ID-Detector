#!/usr/bin/env python3
"""Regenerate matrix-level comparison plots without rerunning inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.benchmarking import generate_comparison_plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate combined benchmark plots.")
    parser.add_argument("--matrix", required=True, help="Matrix folder containing run summaries")
    args = parser.parse_args()
    matrix_dir = Path(args.matrix).resolve()
    if not (matrix_dir / "matrix.json").is_file():
        raise SystemExit(f"matrix.json not found under: {matrix_dir}")
    generated = generate_comparison_plots(matrix_dir)
    matrix_path = matrix_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["comparison_plots"] = generated
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Generated {len(generated)} comparison plot(s) under {matrix_dir / 'comparison_plots'}")


if __name__ == "__main__":
    main()
