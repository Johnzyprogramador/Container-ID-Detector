"""Dependency-free audit and YOLO dataset export utilities."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .schemas import DETECTION_CLASSES, ImageAnnotation

CLASS_TO_ID = {name: index for index, name in enumerate(DETECTION_CLASSES)}


@dataclass(frozen=True)
class AuditResult:
    frame_count: int
    annotation_count: int
    positive_count: int
    negative_count: int
    class_counts: dict[str, int]
    session_counts: dict[str, int]
    missing_images: tuple[str, ...]
    invalid_annotations: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "annotation_count": self.annotation_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "class_counts": self.class_counts,
            "session_counts": self.session_counts,
            "missing_images": list(self.missing_images),
            "invalid_annotations": list(self.invalid_annotations),
        }


def image_path_for_annotation(frames_dir: Path, annotations_dir: Path, path: Path) -> Path:
    relative = path.relative_to(annotations_dir)
    if relative.suffix != ".json":
        raise ValueError(f"Annotation must end in .json: {path}")
    return frames_dir / relative.with_suffix("")


def audit_annotations(frames_dir: Path, annotations_dir: Path) -> AuditResult:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frame_count = sum(
        1 for path in frames_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_extensions
    )
    annotation_paths = sorted(annotations_dir.rglob("*.json"))
    classes: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    positives = 0
    negatives = 0
    missing_images = []
    invalid = []

    for path in annotation_paths:
        try:
            annotation = ImageAnnotation.load(path)
            image_path = image_path_for_annotation(frames_dir, annotations_dir, path)
            if not image_path.is_file():
                missing_images.append(str(image_path))
            sessions[annotation.session_id] += 1
            if annotation.objects:
                positives += 1
            else:
                negatives += 1
            classes.update(item.class_name for item in annotation.objects)
        except Exception as exc:
            invalid.append(f"{path}: {exc}")

    return AuditResult(
        frame_count=frame_count,
        annotation_count=len(annotation_paths),
        positive_count=positives,
        negative_count=negatives,
        class_counts=dict(sorted(classes.items())),
        session_counts=dict(sorted(sessions.items())),
        missing_images=tuple(missing_images),
        invalid_annotations=tuple(invalid),
    )


def export_yolo_dataset(
    *,
    frames_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    validation_sessions: set[str],
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    audit = audit_annotations(frames_dir, annotations_dir)
    if audit.invalid_annotations or audit.missing_images:
        raise ValueError("Audit failed; fix invalid annotations or missing images before export")

    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    included_sessions: dict[str, set[str]] = {"train": set(), "val": set()}

    for annotation_path in sorted(annotations_dir.rglob("*.json")):
        annotation = ImageAnnotation.load(annotation_path)
        split = "val" if annotation.session_id in validation_sessions else "train"
        image_source = image_path_for_annotation(frames_dir, annotations_dir, annotation_path)
        relative = annotation_path.relative_to(annotations_dir).with_suffix("")
        destination_name = "__".join(relative.parts)
        image_destination = output_dir / "images" / split / destination_name
        label_destination = output_dir / "labels" / split / f"{Path(destination_name).stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_destination)

        lines = []
        for item in annotation.objects:
            yolo = item.bbox.as_yolo(image_width=annotation.width, image_height=annotation.height)
            lines.append(f"{CLASS_TO_ID[item.class_name]} " + " ".join(f"{value:.8f}" for value in yolo))
            class_counts[item.class_name] += 1
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""))
        split_counts[split] += 1
        included_sessions[split].add(annotation.session_id)

    if not split_counts["train"] or not split_counts["val"]:
        raise ValueError("Export requires at least one annotated image in both train and val")

    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(DETECTION_CLASSES)],
    ]
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    manifest = {
        "classes": list(DETECTION_CLASSES),
        "class_to_id": CLASS_TO_ID,
        "split_image_counts": dict(split_counts),
        "class_box_counts": dict(sorted(class_counts.items())),
        "sessions": {key: sorted(value) for key, value in included_sessions.items()},
        "source_audit": audit.to_dict(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
