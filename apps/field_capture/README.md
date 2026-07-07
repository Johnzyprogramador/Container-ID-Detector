# Field capture

This local-first service records rolling browser video segments, stores sampled
inference frames, runs the custom YOLO detector on CPU, and records per-frame
timings. The phone UI intentionally exposes only a 1x/2x logical-load selector
and a start/stop button.

Install and run without YOLO to verify camera, recording, storage, and latency:

```bash
pip install -e '.[field]'
python apps/field_capture/app.py --host 127.0.0.1 --port 8080
```

Run CPU inference with the custom detector:

```bash
python apps/field_capture/app.py \
  --host 0.0.0.0 \
  --port 8080 \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt
```

Open `http://127.0.0.1:8080` on the same computer. Camera access from an iPhone
to a LAN IP requires HTTPS; plain HTTP is only suitable for same-device local
testing. Keep the browser page visible and the phone awake while recording.

Each session is stored under `data/field_captures/<session_id>/`:

```text
session.json
cloud_recording/   one-minute MediaRecorder segments
inference_frames/  JPEG samples sent at 2 FPS
results/           YOLO detections and server timings
metrics/frames.csv
```

The browser also keeps each completed video segment in IndexedDB as a local
device backup while browser storage capacity permits. A local-backup failure
does not interrupt cloud recording. Cloud files belong on a mounted persistent
volume when using Docker; never rely on the container writable layer.

Build and run the CPU container:

```bash
docker build -f apps/field_capture/Dockerfile -t container-field-capture .
docker run --rm -p 8080:8080 \
  -v "$PWD/data:/data" \
  -v "$PWD/runs:/models:ro" \
  container-field-capture \
  python apps/field_capture/app.py \
    --host 0.0.0.0 --port 8080 \
    --storage-dir /data/field_captures \
    --weights /models/detection/temporary_two_class_detector/weights/best.pt
```

The initial transport uses periodic JPEG HTTP requests for transparent latency
measurement. Replacing it with WebRTC for the public cloud test will not change
the session layout, detector, or result format.
