#!/usr/bin/env bash
# Run the Flux2 Image Generator server
# Usage: bash run.sh

set -e

# Ensure required directories
mkdir -p outputs loras models

# Install aria2c for fast CivitAI downloads (if not already installed)
if ! command -v aria2c &> /dev/null; then
  echo "Installing aria2 for fast multi-connection downloads..."
  sudo apt-get update -qq && sudo apt-get install -y -qq aria2
fi

# Enable hf_transfer for fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Start the FastAPI server
echo "Starting Flux2 server on ${HOST:-0.0.0.0}:${PORT:-8080} ..."
echo "On first run, models will be downloaded automatically."
echo ""
python -m uvicorn backend.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8080}" \
  --reload
