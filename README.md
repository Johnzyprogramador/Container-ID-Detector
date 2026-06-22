# Container ID Detector

A computer-vision pipeline for locating painted container identifiers,
recognizing their digits, and stabilizing results across video frames.

```text
image/video -> YOLO detection -> crop -> OCR -> tracking -> multi-frame voting
```

## Scope

- Import images and videos as capture sessions.
- Draw and review `painted_number` bounding boxes.
- Store the text inside each box as a transcription.
- Export immutable YOLO and OCR dataset releases.
- Train and evaluate detector and recognizer models.
- Visualize predictions on images and video.
- Add an annotation/training/inference workbench after the core workflows work.

No datasets, model weights, or generated runs belong in Git.

## Layout

```text
apps/workbench/          future annotation and inference UI
configs/                 versioned training and pipeline defaults
docs/                    data format and annotation rules
src/container_vision/    reusable data and vision library
scripts/                 command-line workflows
tests/                   schema and pipeline tests
data/                     datasets (Gitignored)
models/                   model artifacts (Gitignored)
runs/                     training/evaluation runs (Gitignored)
outputs/                  inference results (Gitignored)
```

## Data lifecycle

```text
raw sessions
    -> master annotations
    -> immutable dataset release
    -> versioned model
    -> inference run
    -> human review/correction
```

Raw evidence, verified annotations, generated exports, predictions, and human
corrections remain separate. See [docs/data-format.md](docs/data-format.md).

## Initial development order

1. Finalize the annotation schema and collect representative camera samples.
2. Build folder import and annotation workflows.
3. Export YOLO labels and OCR crops.
4. Train and evaluate a YOLO nano detector.
5. Add a digit-focused recognizer.
6. Add tracking and weighted multi-frame voting.
7. Build the workbench UI on top of the tested library.

## Development

The foundation uses Python 3.11+ and has no runtime dependencies yet:

```bash
python -m unittest discover -s tests
```

Model and UI dependencies will be introduced as optional groups when those
features are implemented.

