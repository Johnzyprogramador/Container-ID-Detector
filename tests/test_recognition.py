from __future__ import annotations

import unittest

from container_vision.recognition import clean_text, parse_easy_result


class RecognitionTests(unittest.TestCase):
    def test_painted_number_cleanup_keeps_digits(self) -> None:
        self.assertEqual(clean_text(" 27-A4 ", "painted_number"), "274")

    def test_license_plate_cleanup_keeps_alphanumeric(self) -> None:
        self.assertEqual(clean_text("ab-12 cd", "license_plate"), "AB12CD")

    def test_parse_easy_result(self) -> None:
        result = parse_easy_result((None, "AB-12", 0.82), "license_plate")
        self.assertEqual(result.text, "AB12")
        self.assertAlmostEqual(result.confidence, 0.82)
