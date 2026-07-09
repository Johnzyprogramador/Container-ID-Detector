FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIELD_CAPTURE_STORAGE_DIR=/data/field_captures \
    MAX_UPLOAD_MB=250

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/field_capture ./apps/field_capture
COPY models ./models
RUN pip install --no-cache-dir -e '.[field]'

EXPOSE 8080
CMD ["sh", "-c", "args=\"--host 0.0.0.0 --port ${PORT:-8080}\"; if [ -n \"$FIELD_CAPTURE_WEIGHTS\" ]; then args=\"$args --weights $FIELD_CAPTURE_WEIGHTS\"; elif [ -f /app/models/field/best.pt ]; then args=\"$args --weights /app/models/field/best.pt\"; fi; exec python apps/field_capture/app.py $args"]
