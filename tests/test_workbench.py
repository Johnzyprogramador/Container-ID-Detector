from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from apps.workbench.app import (
    WorkbenchConfig,
    annotation_from_payload,
    annotation_path_for,
    list_images,
    safe_relative_path,
)


class WorkbenchTests(unittest.TestCase):
    def test_list_images_recursively(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"")
            (root / "nested" / "b.png").write_bytes(b"")
            (root / "notes.txt").write_text("skip")

            images = list_images(root)

            self.assertEqual([item["id"] for item in images], ["a.jpg", "nested/b.png"])

    def test_safe_relative_path_rejects_traversal(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_relative_path(Path(tmp), "../outside.jpg")

    def test_annotation_path_preserves_nested_image_path(self) -> None:
        with TemporaryDirectory() as tmp:
            path = annotation_path_for(Path(tmp), "session/a.jpg")

            self.assertEqual(path, (Path(tmp) / "session" / "a.jpg.json").resolve())

    def test_payload_builds_valid_annotation(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/raw"),
            annotations_dir=Path("data/annotations"),
            session_id="session_001",
            host="127.0.0.1",
            port=7860,
        )

        annotation = annotation_from_payload(
            config,
            {
                "image_id": "truck.jpg",
                "width": 100,
                "height": 80,
                "objects": [
                    {
                        "bbox_xyxy": [10, 20, 40, 50],
                        "transcription": "274",
                        "readable": True,
                        "occluded": False,
                    }
                ],
            },
        )

        self.assertEqual(annotation.session_id, "session_001")
        self.assertEqual(annotation.objects[0].transcription, "274")


if __name__ == "__main__":
    unittest.main()
