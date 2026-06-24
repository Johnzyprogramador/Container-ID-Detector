"""Extract representative image frames from video sessions.

This script uses the system `ffmpeg` command because it is reliable for many
video formats and avoids adding Python package dependencies to the project.

Example:
    python scripts/extract_video_frames.py \
      --input data/raw/session_001 \
      --output data/frames/session_001 \
      --fps 1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def find_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def frame_pattern_for(output_dir: Path, video_path: Path) -> Path:
    stem = video_path.stem.replace(" ", "_")
    return output_dir / f"{stem}_frame_%06d.jpg"


def build_ffmpeg_command(video_path: Path, output_pattern: Path, fps: float, quality: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
        str(output_pattern),
    ]


def extract_video(video_path: Path, output_dir: Path, *, fps: float, quality: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frame_pattern_for(output_dir, video_path)
    before = set(output_dir.glob(f"{video_path.stem.replace(' ', '_')}_frame_*.jpg"))
    subprocess.run(build_ffmpeg_command(video_path, output_pattern, fps, quality), check=True)
    after = set(output_dir.glob(f"{video_path.stem.replace(' ', '_')}_frame_*.jpg"))
    return len(after - before)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image frames from one video or a video folder.")
    parser.add_argument("--input", required=True, help="Video file or folder containing videos.")
    parser.add_argument("--output", required=True, help="Folder where extracted .jpg frames will be written.")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract. Start with 1.0; use 0.5 for every 2 seconds.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="JPEG quality for ffmpeg q:v. Lower is better; 2 is high quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required to extract frames.\n"
            "On Ubuntu/Debian, install it with: sudo apt install ffmpeg"
        )

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    videos = find_videos(input_path)
    if not videos:
        raise SystemExit(f"No videos found under: {input_path}")

    print(f"Found {len(videos)} video(s). Extracting to {output_dir}")
    total = 0
    for video_path in videos:
        count = extract_video(video_path, output_dir, fps=args.fps, quality=args.quality)
        total += count
        print(f"{video_path.name}: {count} new frame(s)")
    print(f"Done. Extracted {total} new frame(s).")


if __name__ == "__main__":
    main()
