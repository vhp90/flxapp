#!/usr/bin/env bash

set -euo pipefail

mkdir -p outputs loras models

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "Starting Flux2 server on ${HOST:-0.0.0.0}:${PORT:-8080}"
echo "AUTO_INITIALIZE_PIPELINE=${AUTO_INITIALIZE_PIPELINE:-true}"
echo "ALLOW_MODEL_DOWNLOADS=${ALLOW_MODEL_DOWNLOADS:-false}"
echo "ENABLE_MOCK_GENERATION=${ENABLE_MOCK_GENERATION:-false}"
echo ""
echo "Tip: in Lightning, set ALLOW_MODEL_DOWNLOADS=true if the Hugging Face assets are not already cached."
echo "Tip: for local UI development, set ENABLE_MOCK_GENERATION=true to avoid model work entirely."
echo ""

UVICORN_ARGS=(
  backend.main:app
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8080}"
)

if [[ "${RELOAD:-0}" == "1" ]]; then
  UVICORN_ARGS+=(--reload)
fi

python -m uvicorn "${UVICORN_ARGS[@]}"
