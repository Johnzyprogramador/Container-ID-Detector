# Workbench

This folder contains the browser UI for:

1. importing and browsing capture sessions;
2. drawing boxes and entering transcriptions;
3. reviewing and correcting annotations;
4. creating dataset releases;
5. launching and monitoring training;
6. comparing predictions with verified labels;
7. running image, video, and live-camera inference.

The workbench must call functions from `container_vision`. Dataset, training,
and inference logic should remain usable without the UI.

## First annotation UI

The first UI is dependency-free and runs with Python only:

```bash
python apps/workbench/app.py \
  --images-dir data/frames \
  --annotations-dir data/annotations/manual \
  --host 0.0.0.0 \
  --port 7860
```

Then open:

```text
http://REMOTE_MACHINE_IP:7860
```

If you are running locally on the same machine, use:

```text
http://127.0.0.1:7860
```

Draw one box around each visible painted number, type the digits inside that
box, and press **Save**. The UI writes one JSON annotation per image using the
canonical project schema.

## Video sessions

The annotation UI labels images, not videos directly. If your raw session is an
`.mp4`, extract frames first:

```bash
python scripts/extract_video_frames.py \
  --input data/raw \
  --output data/frames \
  --fps 1
```

Then launch the UI pointing to the frames root. The UI will show
`session_001`, `session_002`, and other frame folders as selectable sessions:

```bash
python apps/workbench/app.py \
  --images-dir data/frames \
  --annotations-dir data/annotations/manual \
  --host 0.0.0.0 \
  --port 7860
```

There is no filesystem folder picker in this first UI. Choose the root folder
by changing `--images-dir` when starting the server.

Example output:

```text
data/annotations/manual/session_001/truck_001.jpg.json
```

This is deliberately simple. Later workbench milestones can add prediction
overlays, YOLO export buttons, OCR review, video playback, and live camera
streams.
