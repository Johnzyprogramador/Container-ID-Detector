# Data format

## Primary unit: capture session

A session represents one truck visit, one video, or one camera burst.
All frames from a session must stay in the same train/validation/test split.

```text
data/raw/session_20260622_0001/
├── session.json
├── original.mp4
└── frames/
    ├── frame_000001.jpg
    └── frame_000002.jpg
```

Example `session.json`:

```json
{
  "schema_version": "1.0",
  "session_id": "session_20260622_0001",
  "source_type": "video",
  "camera_id": "gate_camera_01",
  "captured_at": "2026-06-22T14:38:40Z",
  "location": "main_gate",
  "notes": ""
}
```

## Master image annotation

The canonical format uses absolute pixel coordinates in `xyxy` order:

```json
{
  "schema_version": "1.0",
  "image_id": "session_20260622_0001/frame_000125",
  "session_id": "session_20260622_0001",
  "file": "frames/frame_000125.jpg",
  "width": 1920,
  "height": 1080,
  "objects": [
    {
      "object_id": "number_001",
      "class_name": "painted_number",
      "bbox_xyxy": [240, 310, 430, 440],
      "transcription": "274",
      "readable": true,
      "occluded": false,
      "review_status": "verified"
    }
  ]
}
```

YOLO learns `painted_number`; it does not learn a separate class for `274`.
The transcription trains and evaluates the recognizer.

Verified negative images have an empty `objects` list. They are valuable for
teaching the detector to ignore graffiti, phone numbers, logos, and unrelated
text.

## Dataset releases

Training never reads directly from changing working annotations. A release is
an immutable snapshot:

```text
data/datasets/container_number_v001/
├── manifest.json
├── splits/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
├── yolo/
└── ocr/
```

The manifest records included sessions, split assignment, counts, schema
version, and export settings.

## Inference runs

Frame predictions and final events are separate:

```json
{
  "event_id": "event_0007",
  "session_id": "session_20260622_0001",
  "track_id": 7,
  "predicted_text": "274",
  "confidence": 0.97,
  "votes": {"274": 12, "214": 1},
  "best_frame_id": "frame_000125",
  "review_status": "unreviewed",
  "reviewed_text": null
}
```

The event should also identify detector, recognizer, pipeline configuration,
and code versions.

