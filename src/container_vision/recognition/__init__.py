"""Painted-number crop preprocessing and recognition."""
"""OCR recognition and text-normalization utilities."""

from .paddle import OCRResult, PaddleTextRecognizer, clean_text, parse_paddle_result

__all__ = ["OCRResult", "PaddleTextRecognizer", "clean_text", "parse_paddle_result"]
