"""Painted-number crop preprocessing and recognition."""
"""OCR recognition and text-normalization utilities."""

from .easy import EasyTextRecognizer, OCRResult, clean_text, parse_easy_result

__all__ = ["EasyTextRecognizer", "OCRResult", "clean_text", "parse_easy_result"]
