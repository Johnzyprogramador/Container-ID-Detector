"""PaddleOCR text-line recognition for YOLO crops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any


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


def parse_paddle_result(result: Any, class_name: str) -> OCRResult:
    payload = result.json if hasattr(result, "json") else result
    content = payload.get("res", payload)
    raw_text = str(content.get("rec_text", ""))
    confidence = float(content.get("rec_score", 0.0) or 0.0)
    return OCRResult(raw_text=raw_text, text=clean_text(raw_text, class_name), confidence=confidence)


class PaddleTextRecognizer:
    def __init__(
        self,
        *,
        device: str,
        model_name: str = "en_PP-OCRv5_mobile_rec",
        batch_size: int = 8,
    ) -> None:
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies with: pip install -e '.[recognition]'") from exc
        self.model = TextRecognition(model_name=model_name, device=device)
        self.batch_size = batch_size

    @staticmethod
    def preprocess(crop):
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        height, width = enhanced.shape[:2]
        scale = max(2.0, 48.0 / max(height, 1))
        resized = cv2.resize(
            enhanced,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    def recognize(self, crops: list, classes: list[str]) -> tuple[list[OCRResult], dict[str, float]]:
        if len(crops) != len(classes):
            raise ValueError("crops and classes must have equal lengths")
        if not crops:
            return [], {"ocr_preprocess_ms": 0.0, "ocr_inference_ms": 0.0, "ocr_postprocess_ms": 0.0}

        started = perf_counter_ns()
        prepared = [self.preprocess(crop) for crop in crops]
        preprocessed = perf_counter_ns()
        raw_results = list(self.model.predict(input=prepared, batch_size=self.batch_size))
        inferred = perf_counter_ns()
        parsed = [parse_paddle_result(result, class_name) for result, class_name in zip(raw_results, classes)]
        finished = perf_counter_ns()
        return parsed, {
            "ocr_preprocess_ms": (preprocessed - started) / 1_000_000,
            "ocr_inference_ms": (inferred - preprocessed) / 1_000_000,
            "ocr_postprocess_ms": (finished - inferred) / 1_000_000,
        }
