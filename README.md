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
apps/workbench/          annotation, prediction, and benchmark UI
apps/field_capture/      mobile browser recording and cloud-latency prototype
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

The foundation uses Python 3.11+ and the first annotation workbench has no
runtime dependencies:

```bash
python -m unittest discover -s tests
```

Launch the annotation UI:

```bash
python apps/workbench/app.py \
  --images-dir data/frames \
  --annotations-dir data/annotations/manual \
  --host 0.0.0.0 \
  --port 7860
```

If your session starts as a video, extract frames first:

```bash
python scripts/extract_video_frames.py \
  --input data/raw \
  --output data/frames \
  --fps 1
```

Model dependencies will be introduced as optional groups when training and
inference features are implemented.

## Export and train the first detector

Audit the local annotations, export a session-safe YOLO release, then train:

```bash
python scripts/audit_annotations.py

python scripts/export_yolo_dataset.py \
  --output data/datasets/two_class_v001 \
  --validation-sessions session_006

pip install -e '.[detection]'

python scripts/train_yolo.py \
  --dataset data/datasets/two_class_v001/data.yaml \
  --model yolo11n.pt \
  --device 0
```

Training outputs are written to this repository's absolute `runs/detection`
directory, independently of any global Ultralytics `runs_dir` setting.

Visualize the trained detector on every original video:

```bash
python scripts/predict_yolo.py \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt \
  --source data/raw \
  --output outputs/predictions \
  --device 0
```

The command searches session folders recursively, saves annotated media while
preserving those folders, and writes aggregate counts to `summary.json`.

## Add OCR to the detected crops

EasyOCR shares PyTorch and CUDA with YOLO, avoiding conflicts between separate
deep-learning runtimes:

```bash
pip install -e '.[detection,recognition]'
```

Then run the combined pipeline:

```bash
python scripts/predict_yolo_ocr.py \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt \
  --source data/raw \
  --output outputs/ocr_predictions \
  --yolo-device 0 \
  --ocr-device gpu:0 \
  --ocr-batch 8
```

The first run downloads EasyOCR's English recognition weights. To view the
generated OCR videos in the workbench, launch it with
`--predictions-dir outputs/ocr_predictions`.

## Four-camera CPU/GPU benchmark

Install the complete benchmark environment:

```bash
pip install -e '.[detection,recognition,benchmark]'
```

Run a short smoke matrix before leaving a long experiment unattended:

```bash
python scripts/run_benchmarks.py \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt \
  --source data/raw \
  --modes cpu gpu \
  --fps 1 \
  --duration 30 \
  --warmup 5 \
  --matrix-id smoke
```

Run the complete ten-experiment matrix (four cameras, 1–5 FPS each, CPU and
GPU, ten minutes per experiment):

```bash
python -u scripts/run_benchmarks.py \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt \
  --source data/raw \
  --modes cpu gpu \
  --fps 1 2 3 4 5 \
  --duration 600 \
  --warmup 60
```

The runner flushes CSV rows continuously, isolates failures to one experiment,
and writes status/error files before continuing. Pass a previous `--matrix-id`
to resume it; completed runs are skipped. Each run receives latency, resource,
event and drop summaries plus PNG plots. The matrix receives CPU-vs-GPU plots.

Launch the workbench with `--benchmarks-dir runs/benchmarks`, open the
**Benchmarks** tab, and leave it open while the command runs. It polls current
progress and resource metrics every five seconds.

Regenerate the combined all-experiment plots from an existing completed matrix
without rerunning inference:

```bash
python scripts/plot_benchmarks.py --matrix runs/benchmarks/MATRIX_ID
```

Combine a baseline and stress matrix into one set of plots:

```bash
python scripts/plot_benchmarks.py \
  --matrix runs/benchmarks/BASELINE_ID runs/benchmarks/STRESS_ID \
  --output runs/benchmarks/combined_capacity
```

The exported release contains copied images, YOLO labels, `data.yaml`, and a
manifest recording classes, counts, and session assignments. Never overwrite a
release used by an experiment; create `two_class_v002` instead.

## Mobile field capture

The local-first field service provides a minimal phone recorder, rolling cloud
video segments, 2 FPS sampled CPU inference, 1x/2x logical-load simulation, and
per-frame latency records. Start in latency-only mode:

```bash
pip install -e '.[field]'
python apps/field_capture/app.py --host 127.0.0.1 --port 8080
```

Or enable the custom CPU YOLO model:

```bash
python apps/field_capture/app.py \
  --host 0.0.0.0 --port 8080 \
  --weights runs/detection/temporary_two_class_detector/weights/best.pt
```

See [apps/field_capture/README.md](apps/field_capture/README.md) for the session
layout, Docker command, Cloud Run deployment commands, local browser limitations,
token auth, ZIP export, and temporary-storage caveats.

## Review Cloud Run field-capture ZIPs

After a factory/cloud test, download each session ZIP from the field-capture
**Sessions** tab and unpack it under `data/field_captures`:

```bash
mkdir -p data/field_captures

for zip in ~/Downloads/container_capture_zips/*.zip; do
  name=$(basename "$zip" .zip)
  mkdir -p "data/field_captures/$name"
  unzip -o "$zip" -d "data/field_captures/$name"
done
```

Then open the main workbench and use the **Field captures** tab:

```bash
python apps/workbench/app.py \
  --host 0.0.0.0 \
  --port 7860 \
  --field-captures-dir data/field_captures
```

The viewer reads `session.json`, `cloud_recording/`, `inference_frames/`,
`metrics/frames.csv`, and `metrics/client.csv`. It works for latency-only runs
and for later YOLO-enabled Cloud Run runs. Latency-only runs will show round-trip
and server/decode timings, but inference and detections will be zero.
