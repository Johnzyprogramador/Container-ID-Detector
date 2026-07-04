#!/usr/bin/env python3
"""Regenerate matrix-level comparison plots without rerunning inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from container_vision.benchmarking import generate_comparison_plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate combined benchmark plots.")
    parser.add_argument(
        "--matrix", nargs="+", required=True,
        help="One or more matrix folders containing run summaries",
    )
    parser.add_argument(
        "--output", default=None,
        help="Combined report folder; required only when passing multiple matrices",
    )
    args = parser.parse_args()
    matrix_dirs = [Path(value).resolve() for value in args.matrix]
    for matrix_dir in matrix_dirs:
        if not (matrix_dir / "matrix.json").is_file():
            raise SystemExit(f"matrix.json not found under: {matrix_dir}")

    if len(matrix_dirs) == 1:
        matrix_dir = matrix_dirs[0]
    else:
        default_name = f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        matrix_dir = Path(args.output).resolve() if args.output else matrix_dirs[0].parent / default_name
        matrix_dir.mkdir(parents=True, exist_ok=True)
        copied_runs = []
        for source_matrix in matrix_dirs:
            for summary_path in sorted(source_matrix.glob("*/summary.json")):
                destination = matrix_dir / f"{source_matrix.name}__{summary_path.parent.name}"
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path, destination / "summary.json")
                shutil.copy2(summary_path, destination / "live.json")
                copied_runs.append(destination.name)
        (matrix_dir / "matrix.json").write_text(json.dumps({
            "matrix_id": matrix_dir.name,
            "status": "complete",
            "source_matrices": [str(path) for path in matrix_dirs],
            "runs": copied_runs,
        }, indent=2) + "\n")

    generated = generate_comparison_plots(matrix_dir)
    matrix_path = matrix_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["comparison_plots"] = generated
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Generated {len(generated)} comparison plot(s) under {matrix_dir / 'comparison_plots'}")


if __name__ == "__main__":
    main()
