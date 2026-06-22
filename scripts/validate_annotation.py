#!/usr/bin/env python3
"""Validate one canonical image-annotation JSON file."""

from __future__ import annotations

import argparse

from container_vision.data import ImageAnnotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation")
    args = parser.parse_args()

    annotation = ImageAnnotation.load(args.annotation)
    print(
        f"Valid: {annotation.image_id} "
        f"({len(annotation.objects)} painted-number object(s))"
    )


if __name__ == "__main__":
    main()

