"""
Auto-download utilities for CivitAI and HuggingFace models.

- CivitAI: Uses aria2c (16 connections) when available, falls back to fast HTTP streaming.
- HuggingFace: Uses huggingface_hub with hf_transfer for fast parallel downloads.
"""

import os
import shutil
import subprocess
import logging
import time
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CivitAI Download
# ---------------------------------------------------------------------------

CIVITAI_API_BASE = "https://civitai.com/api/download/models"


def download_civitai_model(
    model_version_id: str,
    output_path: str,
    token: str | None = None,
) -> Path:
    """
    Download a model file from CivitAI.
    Uses aria2c for max speed (16 connections), falls back to streaming HTTP.
    Skips if file already exists and is non-empty.
    """
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 100_000:
        logger.info("Model already exists at %s (%.1f GB), skipping download.", output, output.stat().st_size / 1e9)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    url = f"{CIVITAI_API_BASE}/{model_version_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Resolve any redirects to get the final URL (needed for aria2c)
    logger.info("Resolving download URL for CivitAI model version %s...", model_version_id)
    try:
        head = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        final_url = head.url
        # Get file size from headers
        content_length = int(head.headers.get("content-length", 0))
        if content_length > 0:
            logger.info("File size: %.2f GB", content_length / 1e9)
    except Exception:
        final_url = url
        if token:
            final_url += f"?token={token}"

    # Try aria2c first (fastest)
    if shutil.which("aria2c"):
        logger.info("Using aria2c for fast multi-connection download...")
        success = _download_aria2c(final_url, output, headers)
        if success:
            return output

    # Try wget
    if shutil.which("wget"):
        logger.info("Using wget for download...")
        success = _download_wget(final_url, output, headers)
        if success:
            return output

    # Fallback to Python requests
    logger.info("Using Python HTTP streaming download...")
    _download_http(url, output, headers)
    return output


def _download_aria2c(url: str, output_path: Path, headers: dict) -> bool:
    """Download using aria2c with 16 connections."""
    try:
        cmd = [
            "aria2c",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=4M",
            "--max-concurrent-downloads=1",
            "--file-allocation=none",
            "--dir", str(output_path.parent),
            "--out", output_path.name,
            "--continue=true",
            "--auto-file-renaming=false",
            "--console-log-level=notice",
            "--summary-interval=10",
        ]
        # Add auth header if present
        for k, v in headers.items():
            cmd.extend(["--header", f"{k}: {v}"])
        cmd.append(url)

        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("aria2c failed: %s", e)
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return False


def _download_wget(url: str, output_path: Path, headers: dict) -> bool:
    """Download using wget."""
    try:
        cmd = [
            "wget",
            "--continue",
            "--progress=bar:force",
            "-O", str(output_path),
        ]
        for k, v in headers.items():
            cmd.extend(["--header", f"{k}: {v}"])
        cmd.append(url)

        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("wget failed: %s", e)
        return False


def _download_http(url: str, output_path: Path, headers: dict, chunk_size: int = 8 * 1024 * 1024):
    """Download using Python requests with streaming. 8MB chunks for speed."""
    tmp_path = output_path.with_suffix(".part")
    downloaded = 0
    if tmp_path.exists():
        downloaded = tmp_path.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + downloaded

        mode = "ab" if downloaded > 0 else "wb"
        start_time = time.time()

        with open(tmp_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = downloaded / elapsed / 1e6
                    if total > 0:
                        pct = downloaded / total * 100
                        logger.info(
                            "Download: %.1f%% (%.1f / %.1f GB) @ %.1f MB/s",
                            pct, downloaded / 1e9, total / 1e9, speed,
                        )

    # Rename .part to final name
    tmp_path.rename(output_path)
    logger.info("Download complete: %s (%.2f GB)", output_path, output_path.stat().st_size / 1e9)


# ---------------------------------------------------------------------------
# HuggingFace Download (for text encoder / VAE / configs)
# ---------------------------------------------------------------------------

def ensure_hf_model_cached(model_id: str, token: str | None = None) -> str:
    """
    Ensure a HuggingFace model is downloaded/cached.
    Returns the model_id (transformers/diffusers will use cached version).
    Uses hf_transfer for speed if available.
    """
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        logger.info("hf_transfer enabled for fast downloads.")
    except ImportError:
        pass

    from huggingface_hub import snapshot_download

    logger.info("Ensuring HuggingFace model is cached: %s", model_id)
    local_dir = snapshot_download(
        model_id,
        token=token,
    )
    logger.info("Model cached at: %s", local_dir)
    return model_id  # transformers/diffusers resolve from cache automatically


def ensure_hf_subfolder_cached(
    repo_id: str,
    subfolder: str,
    token: str | None = None,
) -> str:
    """
    Download only a specific subfolder from a HuggingFace repo.
    Returns the repo_id (caller uses subfolder= when loading).
    """
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    from huggingface_hub import snapshot_download

    logger.info("Ensuring HF subfolder cached: %s/%s", repo_id, subfolder)
    snapshot_download(
        repo_id,
        allow_patterns=[f"{subfolder}/*"],
        token=token,
    )
    return repo_id
