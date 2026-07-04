"""Canonical dependency-free data contracts.

These schemas are deliberately independent from YOLO and OCR libraries. The
annotation UI, exporters, tests, and training code should all use them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
PAINTED_NUMBER_CLASS = "painted_number"
LICENSE_PLATE_CLASS = "license_plate"
DETECTION_CLASSES = (PAINTED_NUMBER_CLASS, LICENSE_PLATE_CLASS)


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class BoundingBox:
    """Absolute pixel coordinates in x1, y1, x2, y2 order."""

    x1: int
    y1: int
    x2: int
    y2: int

    def validate(self, *, image_width: int, image_height: int) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Bounding-box coordinates cannot be negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        if self.x2 > image_width or self.y2 > image_height:
            raise ValueError("Bounding box must remain inside the image")

    def as_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_yolo(self, *, image_width: int, image_height: int) -> tuple[float, ...]:
        """Return normalized x-center, y-center, width, and height."""
        self.validate(image_width=image_width, image_height=image_height)
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return (
            (self.x1 + self.x2) / 2 / image_width,
            (self.y1 + self.y2) / 2 / image_height,
            width / image_width,
            height / image_height,
        )

    @classmethod
    def from_xyxy(cls, value: list[int] | tuple[int, int, int, int]) -> BoundingBox:
        if len(value) != 4:
            raise ValueError("bbox_xyxy must contain exactly four integers")
        return cls(*(int(item) for item in value))


@dataclass(frozen=True)
class AnnotationObject:
    object_id: str
    bbox: BoundingBox
    transcription: str | None
    readable: bool
    occluded: bool = False
    review_status: ReviewStatus = ReviewStatus.DRAFT
    class_name: str = PAINTED_NUMBER_CLASS

    def validate(self, *, image_width: int, image_height: int) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id cannot be empty")
        if self.class_name not in DETECTION_CLASSES:
            raise ValueError(f"class_name must be one of {DETECTION_CLASSES!r}")
        self.bbox.validate(image_width=image_width, image_height=image_height)
        if self.readable:
            if not self.transcription:
                raise ValueError("Readable objects require a transcription")
            if self.class_name == PAINTED_NUMBER_CLASS and not self.transcription.isdigit():
                raise ValueError("Painted-number transcription must contain digits only")
            if self.class_name == LICENSE_PLATE_CLASS and not self.transcription.isalnum():
                raise ValueError("License-plate transcription must contain letters and digits only")
        elif self.transcription:
            raise ValueError("Unreadable objects cannot have a transcription")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "bbox_xyxy": self.bbox.as_xyxy(),
            "transcription": self.transcription,
            "readable": self.readable,
            "occluded": self.occluded,
            "review_status": self.review_status.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnnotationObject:
        return cls(
            object_id=str(value["object_id"]),
            class_name=str(value.get("class_name", PAINTED_NUMBER_CLASS)),
            bbox=BoundingBox.from_xyxy(value["bbox_xyxy"]),
            transcription=value.get("transcription"),
            readable=bool(value["readable"]),
            occluded=bool(value.get("occluded", False)),
            review_status=ReviewStatus(value.get("review_status", ReviewStatus.DRAFT)),
        )


@dataclass(frozen=True)
class ImageAnnotation:
    image_id: str
    session_id: str
    file: str
    width: int
    height: int
    objects: tuple[AnnotationObject, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.image_id.strip() or not self.session_id.strip():
            raise ValueError("image_id and session_id cannot be empty")
        if not self.file.strip():
            raise ValueError("file cannot be empty")
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object_id values must be unique within an image")
        for item in self.objects:
            item.validate(image_width=self.width, image_height=self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "session_id": self.session_id,
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "objects": [item.to_dict() for item in self.objects],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImageAnnotation:
        annotation = cls(
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
            image_id=str(value["image_id"]),
            session_id=str(value["session_id"]),
            file=str(value["file"]),
            width=int(value["width"]),
            height=int(value["height"]),
            objects=tuple(AnnotationObject.from_dict(item) for item in value.get("objects", [])),
        )
        annotation.validate()
        return annotation

    @classmethod
    def load(cls, path: str | Path) -> ImageAnnotation:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(frozen=True)
class Session:
    session_id: str
    source_type: str
    camera_id: str | None = None
    captured_at: str | None = None
    location: str | None = None
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if self.source_type not in {"image", "image_folder", "video", "camera"}:
            raise ValueError("Unsupported source_type")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Session:
        session = cls(**value)
        session.validate()
        return session
