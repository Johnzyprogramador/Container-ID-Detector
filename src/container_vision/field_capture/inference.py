from __future__ import annotations

import io
import threading
import time
from pathlib import Path


class CpuYoloDetector:
    """Lazy CPU-only YOLO wrapper; no weights means latency-only capture mode."""

    def __init__(self, weights: Path | None, *, confidence: float = 0.25, image_size: int = 640):
        self.weights = weights
        self.confidence = confidence
        self.image_size = image_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.weights is not None

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install detection dependencies: pip install -e '.[detection]'") from exc
        self._model = YOLO(str(self.weights))

    def predict(self, jpeg: bytes, logical_streams: int) -> tuple[list[dict], float, float]:
        decode_started = time.perf_counter()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for field capture") from exc
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        decode_ms = (time.perf_counter() - decode_started) * 1000
        if not self.enabled:
            return [], decode_ms, 0.0

        self.load()
        inference_started = time.perf_counter()
        result_sets = []
        with self._lock:
            for logical_index in range(logical_streams):
                prediction = self._model.predict(
                    source=image,
                    device="cpu",
                    conf=self.confidence,
                    imgsz=self.image_size,
                    verbose=False,
                )[0]
                result_sets.append((logical_index + 1, prediction))
        inference_ms = (time.perf_counter() - inference_started) * 1000

        detections = []
        # Save one result set as the visible output. Repeated sets represent load only.
        logical_index, prediction = result_sets[0]
        names = prediction.names
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "logical_stream": logical_index,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(float(box.conf.item()), 5),
                    "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )
        return detections, decode_ms, inference_ms
