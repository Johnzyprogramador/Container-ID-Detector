from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.predict_yolo import find_media, output_path_for


class PredictYoloTests(unittest.TestCase):
    def test_find_media_recursively(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "one.mp4").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "two.jpg").write_bytes(b"")
            (root / "notes.txt").write_text("ignore")
            self.assertEqual(
                [path.name for path in find_media(root)],
                ["one.mp4", "two.jpg"],
            )

    def test_output_path_preserves_session(self) -> None:
        source = Path("/data/raw")
        media = source / "session_006" / "truck.mp4"
        destination = output_path_for(source, media, Path("/outputs"))
        self.assertEqual(destination, Path("/outputs/session_006/truck_predicted.mp4"))


if __name__ == "__main__":
    unittest.main()
