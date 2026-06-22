# Architecture

The system is a pipeline of replaceable components rather than one large
model:

```text
source
  -> frame reader
  -> number-region detector
  -> tracker
  -> padded crop
  -> digit recognizer
  -> per-track weighted voting
  -> event result
```

## Boundaries

- `data` owns schemas, annotation storage, dataset releases, and exports.
- `detection` owns number-region localization.
- `recognition` owns crop preprocessing and transcription.
- `tracking` associates detections across frames.
- `voting` turns noisy frame readings into one stable result.
- `pipelines` orchestrates components without implementing their internals.
- `visualization` draws boxes, text, confidence, and track state.
- `apps/workbench` is an interface over the library, not the home of model
  logic.

## Planned interfaces

The first implementations should preserve these conceptual interfaces:

```python
detections = detector.detect(frame)
tracks = tracker.update(detections)
reading = recognizer.read(crop)
event = voter.update(track_id, reading)
```

Keeping these boundaries makes it possible to replace YOLO, OCR, or the
tracker without redesigning the annotation and evaluation system.

## Repository policy

- Raw inputs are immutable.
- Master annotations are the source of truth.
- YOLO labels and OCR crops are generated exports.
- Dataset releases are immutable.
- Every model identifies its dataset release and code revision.
- Predictions never silently overwrite annotations.
- Human corrections are recorded explicitly.

