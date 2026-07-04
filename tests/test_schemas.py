from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from container_vision.data import (
    AnnotationObject,
    BoundingBox,
    ImageAnnotation,
    LICENSE_PLATE_CLASS,
    ReviewStatus,
    Session,
)


class BoundingBoxTests(unittest.TestCase):
    def test_yolo_conversion(self) -> None:
        box = BoundingBox(20, 10, 60, 50)
        self.assertEqual(box.as_yolo(image_width=100, image_height=100), (0.4, 0.3, 0.4, 0.4))

    def test_rejects_box_outside_image(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(20, 10, 120, 50).validate(image_width=100, image_height=100)


class AnnotationTests(unittest.TestCase):
    def make_annotation(self) -> ImageAnnotation:
        return ImageAnnotation(
            image_id="session_1/frame_1",
            session_id="session_1",
            file="frames/frame_1.jpg",
            width=1920,
            height=1080,
            objects=(
                AnnotationObject(
                    object_id="number_1",
                    bbox=BoundingBox(240, 310, 430, 440),
                    transcription="027",
                    readable=True,
                    review_status=ReviewStatus.VERIFIED,
                ),
            ),
        )

    def test_round_trip(self) -> None:
        annotation = self.make_annotation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            annotation.save(path)
            loaded = ImageAnnotation.load(path)
        self.assertEqual(loaded, annotation)

    def test_preserves_leading_zero(self) -> None:
        annotation = self.make_annotation()
        self.assertEqual(annotation.objects[0].transcription, "027")

    def test_readable_requires_digits(self) -> None:
        annotation = ImageAnnotation(
            image_id="image",
            session_id="session",
            file="image.jpg",
            width=100,
            height=100,
            objects=(
                AnnotationObject(
                    object_id="number",
                    bbox=BoundingBox(1, 1, 50, 50),
                    transcription="27A",
                    readable=True,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            annotation.validate()

    def test_license_plate_allows_letters_and_digits(self) -> None:
        annotation = ImageAnnotation(
            image_id="plate_image",
            session_id="session",
            file="plate.jpg",
            width=100,
            height=100,
            objects=(
                AnnotationObject(
                    object_id="plate",
                    class_name=LICENSE_PLATE_CLASS,
                    bbox=BoundingBox(1, 1, 80, 40),
                    transcription="12AB34",
                    readable=True,
                ),
            ),
        )
        annotation.validate()

    def test_verified_negative_is_valid(self) -> None:
        annotation = ImageAnnotation(
            image_id="negative",
            session_id="session",
            file="negative.jpg",
            width=100,
            height=100,
        )
        annotation.validate()

    def test_from_dictionary(self) -> None:
        value = json.loads(json.dumps(self.make_annotation().to_dict()))
        self.assertEqual(ImageAnnotation.from_dict(value), self.make_annotation())


class SessionTests(unittest.TestCase):
    def test_supported_video_session(self) -> None:
        Session(session_id="visit_1", source_type="video").validate()

    def test_rejects_unknown_source(self) -> None:
        with self.assertRaises(ValueError):
            Session(session_id="visit_1", source_type="telepathy").validate()


if __name__ == "__main__":
    unittest.main()
