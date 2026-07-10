# Mirrors the root Dockerfile (used by the Cloud Run deploy). This copy exists
# for local builds: docker build -f apps/field_capture/Dockerfile -t ... .
# Keep the two in sync.
#
# Layer order is deliberate: expensive, rarely-changing dependency layers first
# so caching can reuse them; app code + model copied last so swapping
# models/field/best.pt rebuilds only tiny layers.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FIELD_CAPTURE_STORAGE_DIR=/data/field_captures \
    MAX_UPLOAD_MB=250

WORKDIR /app

# 1) CPU-only PyTorch (avoids ~2-3 GB of unused CUDA libraries on Linux).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) Remaining dependencies (torch already satisfied, so it is not re-pulled).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e '.[field]'

# 3) App code and model last.
COPY apps/field_capture ./apps/field_capture
COPY models ./models

EXPOSE 8080
CMD ["sh", "-c", "args=\"--host 0.0.0.0 --port ${PORT:-8080}\"; if [ -n \"$FIELD_CAPTURE_WEIGHTS\" ]; then args=\"$args --weights $FIELD_CAPTURE_WEIGHTS\"; elif [ -f /app/models/field/best.pt ]; then args=\"$args --weights /app/models/field/best.pt\"; fi; exec python apps/field_capture/app.py $args"]
