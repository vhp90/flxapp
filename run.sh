#!/usr/bin/env bash
# Run the Flux2 Image Generator server
# Usage: bash run.sh

set -e

# Ensure required directories
mkdir -p outputs loras models

# Start the FastAPI server
echo "Starting Flux2 server on ${HOST:-0.0.0.0}:${PORT:-8080} ..."
python -m uvicorn backend.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8080}" \
  --reload
