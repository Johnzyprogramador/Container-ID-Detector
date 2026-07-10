# Field capture

This local-first service records rolling browser video segments, stores sampled
inference frames, runs the custom YOLO detector on CPU, and records per-frame
timings. The phone UI intentionally exposes only a 1x/2x logical-load selector
and a start/stop button.

Cloud Run can host this same service over the managed `https://*.run.app` URL.
No Go backend is required; the Docker container can run the existing Python app.

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
metrics/client.csv
```

The browser also keeps each completed video segment in IndexedDB as a local
device backup while browser storage capacity permits. A local-backup failure
does not interrupt cloud recording. Cloud files belong on a mounted persistent
volume when using Docker; never rely on the container writable layer.

The **Sessions** tab exposes a ZIP download for every capture session. Download
the ZIP immediately after a Cloud Run field test when not using Cloud Storage:
Cloud Run's writable filesystem is temporary and can disappear when the
instance restarts, scales down, or is replaced.

Build and run the CPU container:

```bash
mkdir -p models/field
# Optional: copy your trained detector before building.
# cp runs/detection/temporary_two_class_detector/weights/best.pt models/field/best.pt

docker build -f apps/field_capture/Dockerfile -t container-field-capture .
docker run --rm -p 8080:8080 -v "$PWD/data:/data" container-field-capture
```

If `models/field/best.pt` exists during the build, the container automatically
uses it. Otherwise it runs in latency-only mode. You can also override the model
path with `FIELD_CAPTURE_WEIGHTS`.

Enable a simple field-test token:

```bash
docker run --rm -p 8080:8080 \
  -e FIELD_CAPTURE_TOKEN="change-me-long-random-token" \
  -v "$PWD/data:/data" \
  container-field-capture
```

The initial transport uses periodic JPEG HTTP requests for transparent latency
measurement. Replacing it with WebRTC for the public cloud test will not change
the session layout, detector, or result format.

## Cloud Run deployment

Cloud Run provides HTTPS automatically on the generated `*.run.app` URL. The
container should still listen with normal HTTP internally; Cloud Run terminates
TLS before forwarding traffic to the app.

### Simplest path (one command)

`scripts/deploy_field_capture.sh` builds on Cloud Build and deploys in one step:

```bash
# Latency-only (no detector model):
./scripts/deploy_field_capture.sh teste-501920

# With the real detector — copies your weights into place, then deploys:
./scripts/deploy_field_capture.sh teste-501920 --model /path/to/your/best.pt
```

See `models/field/README.md` for how to add the detector model. The manual
Artifact Registry steps below remain available if you prefer them.

Build with your local model baked into the image:

```bash
mkdir -p models/field
cp runs/detection/temporary_two_class_detector/weights/best.pt models/field/best.pt
docker build -f apps/field_capture/Dockerfile -t container-field-capture .
```

Push the image to Artifact Registry:

```bash
gcloud artifacts repositories create container-vision \
  --repository-format=docker \
  --location=europe-west1

docker tag container-field-capture \
  europe-west1-docker.pkg.dev/PROJECT_ID/container-vision/field-capture:latest

docker push \
  europe-west1-docker.pkg.dev/PROJECT_ID/container-vision/field-capture:latest
```

Deploy one controlled CPU instance:

```bash
gcloud run deploy container-field-capture \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/container-vision/field-capture:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --concurrency 4 \
  --max-instances 1 \
  --timeout 3600 \
  --set-env-vars FIELD_CAPTURE_TOKEN=change-me-long-random-token
```

Open the service on the iPhone with:

```text
https://SERVICE_NAME-xxxxx-ew.a.run.app/?token=change-me-long-random-token
```

The token is stored in the browser for the session so the URL is cleaned after
loading. For a quick private test you can omit `FIELD_CAPTURE_TOKEN`, but a
public unauthenticated field endpoint is not recommended.

Cloud Run local storage is suitable only as a temporary handoff. After stopping
the recording, open **Sessions** and download the ZIP immediately. The phone's
local video backup remains the second safety net. Use Cloud Storage later when
the capture data must persist without manual export.
