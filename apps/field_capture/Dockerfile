FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/field_capture ./apps/field_capture
RUN pip install --no-cache-dir -e '.[field]'

EXPOSE 8080
CMD ["python", "apps/field_capture/app.py", "--host", "0.0.0.0", "--port", "8080", "--storage-dir", "/data/field_captures"]
