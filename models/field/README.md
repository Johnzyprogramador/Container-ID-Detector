# Field detector model

Drop your trained YOLO weights here as **`best.pt`** to turn on real detection.
Until a file named `best.pt` exists here, the service runs in *latency-only*
mode (it measures the full camera→server round-trip but returns no detections).

No code changes are needed — the container auto-detects `models/field/best.pt`
at startup and switches to real detection.

## Easiest way (one line)

Point the deploy script at your weights. It copies them here and deploys:

```bash
./scripts/deploy_field_capture.sh teste-501920 --model /path/to/your/best.pt
```

## Manual way

```bash
cp /path/to/your/best.pt models/field/best.pt
./scripts/deploy_field_capture.sh teste-501920
```

## Notes

- `best.pt` is intentionally **not** committed to git (model weights are large).
  Each developer supplies their own copy locally; it is uploaded to Cloud Build
  at deploy time (allowed through `.gcloudignore`).
- Prerequisites: the `gcloud` CLI installed and authenticated
  (`gcloud auth login`) with deploy access to the target project.
- To confirm which mode is live after deploying, open `<service-url>/api/health`
  — `"inference_enabled": true` means the model is active.
