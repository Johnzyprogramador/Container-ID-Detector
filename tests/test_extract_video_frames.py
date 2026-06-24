from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extract_video_frames import (
    build_ffmpeg_command,
    find_videos,
    frame_pattern_for,
    output_dir_for_video,
)


class ExtractVideoFramesTests(unittest.TestCase):
    def test_find_videos_recursively(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "a.MP4").write_bytes(b"")
            (root / "session_001" / "ignore.jpg").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "b.mov").write_bytes(b"")

            videos = find_videos(root)

            self.assertEqual([item.name for item in videos], ["a.MP4", "b.mov"])

    def test_frame_pattern_uses_video_stem(self) -> None:
        pattern = frame_pattern_for(Path("data/frames/session_001"), Path("my video.mp4"))

        self.assertEqual(pattern.as_posix(), "data/frames/session_001/my_video_frame_%06d.jpg")

    def test_output_dir_preserves_session_when_input_is_root(self) -> None:
        output_dir = output_dir_for_video(
            input_path=Path("data/raw"),
            output_root=Path("data/frames"),
            video_path=Path("data/raw/session_002/video.mp4"),
        )

        self.assertEqual(output_dir.as_posix(), "data/frames/session_002")

    def test_output_dir_uses_given_output_when_input_is_session(self) -> None:
        output_dir = output_dir_for_video(
            input_path=Path("data/raw/session_002"),
            output_root=Path("data/frames/session_002"),
            video_path=Path("data/raw/session_002/video.mp4"),
        )

        self.assertEqual(output_dir.as_posix(), "data/frames/session_002")

    def test_build_ffmpeg_command(self) -> None:
        command = build_ffmpeg_command(Path("input.mp4"), Path("frame_%06d.jpg"), 1.0, 2)

        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "input.mp4",
                "-vf",
                "fps=1.0",
                "-q:v",
                "2",
                "frame_%06d.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
