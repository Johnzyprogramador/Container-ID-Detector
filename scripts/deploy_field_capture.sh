#!/usr/bin/env bash
#
# One-command Cloud Run deploy for the mobile field-capture service.
#
# Builds on Cloud Build with cross-build Docker layer caching (see
# cloudbuild.yaml), then deploys the image to Cloud Run. The first deploy is a
# full build (~3-4 min); later deploys reuse cached layers, so a model-only or
# code-only change is quick.
#
# Prerequisites (one-time):
#   1. A Google Cloud account with a billing account (created in the web
#      console -- the CLI cannot enter a payment method).
#   2. The gcloud CLI installed and authenticated:  gcloud auth login
#
# Usage:
#   scripts/deploy_field_capture.sh PROJECT_ID [REGION] [--model PATH]
#
# Arguments:
#   PROJECT_ID     Google Cloud project ID (required).
#   REGION         Cloud Run region (default: europe-west1).
#
# Options:
#   --model PATH   Copy trained YOLO weights (a .pt file) into
#                  models/field/best.pt before building, so real detection is
#                  baked into the image. Omit for latency-only mode (or, if
#                  models/field/best.pt already exists, it is used as-is).
#
# Environment:
#   FIELD_CAPTURE_TOKEN   Auth token. A fresh one is generated if unset.

set -euo pipefail

usage() {
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="container-field-capture"
AR_REPO="cloud-run-source-deploy"
MODEL_DEST="${REPO_ROOT}/models/field/best.pt"

PROJECT_ID=""
REGION="europe-west1"
MODEL_SRC=""
seen_positional=0

# Parse args. --model takes a value; the first two bare args are PROJECT_ID and
# REGION. Written for bash 3.2 (macOS default) -- no arrays.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_SRC="${2:?--model requires a path}"; shift 2 ;;
    --model=*) MODEL_SRC="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ $seen_positional -eq 0 ]]; then PROJECT_ID="$1"; else REGION="$1"; fi
      seen_positional=$((seen_positional + 1)); shift ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID is required." >&2
  usage
  exit 1
fi

# If a model was provided, copy it into the build location.
if [[ -n "${MODEL_SRC}" ]]; then
  if [[ ! -f "${MODEL_SRC}" ]]; then
    echo "ERROR: model file not found: ${MODEL_SRC}" >&2
    exit 1
  fi
  mkdir -p "${REPO_ROOT}/models/field"
  cp "${MODEL_SRC}" "${MODEL_DEST}"
  echo ">> Copied model -> models/field/best.pt (from ${MODEL_SRC})"
fi

# Generate a token if the caller did not supply one.
if [[ -z "${FIELD_CAPTURE_TOKEN:-}" ]]; then
  FIELD_CAPTURE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo ">> Generated FIELD_CAPTURE_TOKEN: ${FIELD_CAPTURE_TOKEN}"
fi

if [[ -f "${MODEL_DEST}" ]]; then
  echo ">> Detection: REAL — baking in models/field/best.pt ($(du -h "${MODEL_DEST}" | cut -f1))"
else
  echo ">> Detection: latency-only (no models/field/best.pt present)"
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE}"

echo ">> Project : ${PROJECT_ID}"
echo ">> Region  : ${REGION}"
echo ">> Service : ${SERVICE}"
echo ">> Image   : ${IMAGE}:latest"

gcloud config set project "${PROJECT_ID}" >/dev/null

echo ">> Enabling required APIs (run, cloudbuild, artifactregistry)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

echo ">> Ensuring Artifact Registry repo '${AR_REPO}' exists in ${REGION}..."
gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker --location="${REGION}"

echo ">> Building on Cloud Build (reusing cached layers where possible)..."
gcloud builds submit "${REPO_ROOT}" \
  --region="${REGION}" \
  --config="${REPO_ROOT}/cloudbuild.yaml" \
  --substitutions=_IMAGE="${IMAGE}"

echo ">> Deploying image to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --concurrency 4 \
  --max-instances 1 \
  --timeout 3600 \
  --set-env-vars "FIELD_CAPTURE_TOKEN=${FIELD_CAPTURE_TOKEN}"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo
echo "=================================================================="
echo " Deployed. Open this on the phone (token auto-clears from the URL):"
echo
echo "   ${URL}/?token=${FIELD_CAPTURE_TOKEN}"
echo
echo " Health check:  ${URL}/api/health"
echo "=================================================================="
