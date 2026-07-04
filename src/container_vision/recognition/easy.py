"""EasyOCR recognition for YOLO crops using the same PyTorch runtime as YOLO."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class OCRResult:
    raw_text: str
    text: str
    confidence: float


def clean_text(value: str, class_name: str) -> str:
    upper = value.upper()
    if class_name == "painted_number":
        return re.sub(r"[^0-9]", "", upper)
    if class_name == "license_plate":
        return re.sub(r"[^A-Z0-9]", "", upper)
    raise ValueError(f"Unsupported detection class: {class_name}")


def parse_easy_result(result, class_name: str) -> OCRResult:
    raw_text = str(result[1]) if len(result) > 1 else ""
    confidence = float(result[2]) if len(result) > 2 else 0.0
    return OCRResult(raw_text=raw_text, text=clean_text(raw_text, class_name), confidence=confidence)


class EasyTextRecognizer:
    def __init__(self, *, device: str, batch_size: int = 8) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies with: pip install -e '.[recognition]'") from exc
        gpu = device != "cpu"
        self.reader = easyocr.Reader(["en"], gpu=gpu, detector=False, recognizer=True)
        self.batch_size = batch_size

    @staticmethod
    def preprocess(crop):
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        height, width = enhanced.shape[:2]
        scale = 48.0 / max(height, 1)
        return cv2.resize(
            enhanced,
            (max(1, int(width * scale)), 48),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def make_batch_canvas(prepared: list):
        import numpy as np

        gap = 4
        width = max(image.shape[1] for image in prepared)
        height = sum(image.shape[0] + gap for image in prepared) - gap
        canvas = np.full((height, width), 255, dtype=np.uint8)
        boxes = []
        y = 0
        for image in prepared:
            image_height, image_width = image.shape[:2]
            canvas[y:y + image_height, :image_width] = image
            boxes.append([0, image_width, y, y + image_height])
            y += image_height + gap
        return canvas, boxes

    def recognize(self, crops: list, classes: list[str]) -> tuple[list[OCRResult], dict[str, float]]:
        if len(crops) != len(classes):
            raise ValueError("crops and classes must have equal lengths")
        if not crops:
            return [], {"ocr_preprocess_ms": 0.0, "ocr_inference_ms": 0.0, "ocr_postprocess_ms": 0.0}

        started = perf_counter_ns()
        prepared = [self.preprocess(crop) for crop in crops]
        canvas, boxes = self.make_batch_canvas(prepared)
        preprocessed = perf_counter_ns()
        raw_results = self.reader.recognize(
            canvas,
            horizontal_list=boxes,
            free_list=[],
            batch_size=self.batch_size,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            reformat=False,
        )
        inferred = perf_counter_ns()
        parsed = [parse_easy_result(result, class_name) for result, class_name in zip(raw_results, classes)]
        if len(parsed) < len(classes):
            parsed.extend(OCRResult("", "", 0.0) for _ in range(len(classes) - len(parsed)))
        finished = perf_counter_ns()
        return parsed, {
            "ocr_preprocess_ms": (preprocessed - started) / 1_000_000,
            "ocr_inference_ms": (inferred - preprocessed) / 1_000_000,
            "ocr_postprocess_ms": (finished - inferred) / 1_000_000,
        }
