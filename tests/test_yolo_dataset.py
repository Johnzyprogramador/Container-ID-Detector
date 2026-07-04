from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from container_vision.data import AnnotationObject, BoundingBox, ImageAnnotation
from container_vision.data.yolo import audit_annotations, export_yolo_dataset


class YoloDatasetTests(unittest.TestCase):
    def save_example(self, root: Path, session: str, name: str, objects=()) -> None:
        frame = root / "frames" / session / name
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"image")
        annotation = ImageAnnotation(
            image_id=f"{session}/{name}",
            session_id=session,
            file=f"{session}/{name}",
            width=100,
            height=100,
            objects=tuple(objects),
        )
        annotation.save(root / "annotations" / session / f"{name}.json")

    def test_audit_counts_positive_and_negative_images(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_example(
                root,
                "session_001",
                "positive.jpg",
                [AnnotationObject("box", BoundingBox(10, 20, 50, 60), "274", True)],
            )
            self.save_example(root, "session_002", "negative.jpg")
            result = audit_annotations(root / "frames", root / "annotations")
            self.assertEqual(result.frame_count, 2)
            self.assertEqual(result.positive_count, 1)
            self.assertEqual(result.negative_count, 1)
            self.assertEqual(result.class_counts, {"painted_number": 1})

    def test_export_preserves_session_split_and_negative_label(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_example(
                root,
                "session_train",
                "positive.jpg",
                [AnnotationObject("box", BoundingBox(10, 20, 50, 60), "274", True)],
            )
            self.save_example(root, "session_val", "negative.jpg")
            output = root / "dataset"
            manifest = export_yolo_dataset(
                frames_dir=root / "frames",
                annotations_dir=root / "annotations",
                output_dir=output,
                validation_sessions={"session_val"},
            )
            self.assertEqual(manifest["split_image_counts"], {"train": 1, "val": 1})
            self.assertTrue((output / "images/train/session_train__positive.jpg").is_file())
            self.assertEqual((output / "labels/val/session_val__negative.txt").read_text(), "")
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
