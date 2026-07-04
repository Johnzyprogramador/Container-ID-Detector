from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from apps.workbench.app import (
    WorkbenchConfig,
    annotation_from_payload,
    annotation_path_for,
    list_images,
    list_prediction_videos,
    list_sessions,
    safe_relative_path,
    session_id_from_image_id,
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

    def test_list_sessions_from_frames_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "a.jpg").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "b.png").write_bytes(b"")
            (root / "empty_session").mkdir()

            sessions = list_sessions(root)

            self.assertEqual([item["id"] for item in sessions], ["session_001", "session_002"])
            self.assertEqual([item["image_count"] for item in sessions], [1, 1])

    def test_list_images_for_selected_session(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "a.jpg").write_bytes(b"")
            (root / "session_002").mkdir()
            (root / "session_002" / "b.png").write_bytes(b"")

            images = list_images(root, session_id="session_002")

            self.assertEqual([item["id"] for item in images], ["session_002/b.png"])
            self.assertEqual([item["path"] for item in images], ["b.png"])

    def test_list_prediction_videos_by_session(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_001").mkdir()
            (root / "session_001" / "truck_predicted.mp4").write_bytes(b"")
            (root / "summary.json").write_text("{}")

            videos = list_prediction_videos(root)

            self.assertEqual(len(videos), 1)
            self.assertEqual(videos[0]["session"], "session_001")
            self.assertEqual(videos[0]["url"], "/prediction-media/session_001/truck_predicted.mp4")

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
            images_dir=Path("data/frames"),
            annotations_dir=Path("data/annotations"),
            session_id=None,
            host="127.0.0.1",
            port=7860,
        )

        annotation = annotation_from_payload(
            config,
            {
                "image_id": "session_001/truck.jpg",
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
        self.assertEqual(annotation.objects[0].class_name, "painted_number")
        self.assertEqual(annotation.objects[0].transcription, "274")

    def test_payload_preserves_license_plate_class(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/frames"),
            annotations_dir=Path("data/annotations"),
            session_id=None,
            host="127.0.0.1",
            port=7860,
        )

        annotation = annotation_from_payload(
            config,
            {
                "image_id": "session_plate/car.jpg",
                "width": 100,
                "height": 80,
                "objects": [
                    {
                        "class_name": "license_plate",
                        "bbox_xyxy": [10, 20, 70, 50],
                        "transcription": "12AB34",
                        "readable": True,
                    }
                ],
            },
        )

        annotation.validate()
        self.assertEqual(annotation.objects[0].class_name, "license_plate")
        self.assertEqual(annotation.objects[0].transcription, "12AB34")

    def test_forced_session_id_overrides_image_path_session(self) -> None:
        config = WorkbenchConfig(
            images_dir=Path("data/frames/session_001"),
            annotations_dir=Path("data/annotations/session_001"),
            session_id="session_001",
            host="127.0.0.1",
            port=7860,
        )

        self.assertEqual(session_id_from_image_id(config, "truck.jpg"), "session_001")


if __name__ == "__main__":
    unittest.main()
