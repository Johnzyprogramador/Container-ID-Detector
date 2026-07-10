# Root Dockerfile used by scripts/deploy_field_capture.sh via cloudbuild.yaml.
# Mirrors apps/field_capture/Dockerfile — both build from the repo root as
# context. Keep the two in sync.
#
# Layer order is deliberate: the expensive, rarely-changing dependency layers
# come first so Cloud Build's --cache-from can reuse them, and app code + model
# are copied last so swapping models/field/best.pt rebuilds only tiny layers.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FIELD_CAPTURE_STORAGE_DIR=/data/field_captures \
    MAX_UPLOAD_MB=250

WORKDIR /app

# 1) CPU-only PyTorch. The default torch wheel on Linux pulls ~2-3 GB of CUDA
#    libraries we never use on Cloud Run (CPU-only). The CPU index avoids that,
#    cutting build time, image size, and cold start. Changes rarely -> cached.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) Remaining dependencies. Cached until pyproject.toml or src change. torch is
#    already satisfied above, so ultralytics does not re-pull it.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e '.[field]'

# 3) App code and model LAST. Swapping models/field/best.pt or editing the app
#    rebuilds only these layers, not the dependency install above.
COPY apps/field_capture ./apps/field_capture
COPY models ./models

EXPOSE 8080
CMD ["sh", "-c", "args=\"--host 0.0.0.0 --port ${PORT:-8080}\"; if [ -n \"$FIELD_CAPTURE_WEIGHTS\" ]; then args=\"$args --weights $FIELD_CAPTURE_WEIGHTS\"; elif [ -f /app/models/field/best.pt ]; then args=\"$args --weights /app/models/field/best.pt\"; fi; exec python apps/field_capture/app.py $args"]
