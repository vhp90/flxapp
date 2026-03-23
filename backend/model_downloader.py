"""
Auto-download utilities for CivitAI and HuggingFace models.

- CivitAI: Uses aria2c (multi-connection) when available, falls back to fast HTTP streaming.
- HuggingFace: Uses huggingface_hub with hf_transfer for fast parallel downloads.
- Parallel: All model components download simultaneously via ThreadPoolExecutor.
"""

import os
import shutil
import subprocess
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_token(token: str | None) -> str | None:
    """Convert empty/whitespace-only tokens to None to avoid 'Bearer ' header errors."""
    if token is None:
        return None
    token = token.strip()
    return token if token else None


def _enable_hf_transfer():
    """Enable hf_transfer for Rust-based parallel chunked downloads."""
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        return True
    except ImportError:
        return False


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
    Uses aria2c for max speed, falls back to streaming HTTP.
    Skips if file already exists and is non-empty.
    """
    token = _normalize_token(token)
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 100_000:
        logger.info("Model already exists at %s (%.1f GB), skipping download.", output, output.stat().st_size / 1e9)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    # CivitAI uses ?token= query param, NOT Bearer header
    url = f"{CIVITAI_API_BASE}/{model_version_id}"
    if token:
        url += f"?token={token}"

    # Resolve any redirects to get the final CDN URL (needed for aria2c)
    logger.info("Resolving download URL for CivitAI model version %s...", model_version_id)
    final_url = url
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        if head.status_code == 200:
            final_url = head.url
            content_length = int(head.headers.get("content-length", 0))
            if content_length > 0:
                logger.info("File size: %.2f GB", content_length / 1e9)
        else:
            logger.warning("HEAD request returned %s, using original URL", head.status_code)
    except Exception as e:
        logger.warning("Could not resolve URL: %s, using original", e)

    # Try aria2c first (fastest — multi-connection to CDN)
    if shutil.which("aria2c"):
        logger.info("Using aria2c for fast multi-connection download...")
        success = _download_aria2c(final_url, output)
        if success:
            return output

    # Try wget
    if shutil.which("wget"):
        logger.info("Using wget for download...")
        success = _download_wget(final_url, output)
        if success:
            return output

    # Fallback to Python requests
    logger.info("Using Python HTTP streaming download...")
    _download_http(final_url, output)
    return output


def _download_aria2c(url: str, output_path: Path) -> bool:
    """Download using aria2c with multi-connection + retries."""
    try:
        cmd = [
            "aria2c",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=8M",
            "--max-concurrent-downloads=1",
            "--file-allocation=none",
            "--dir", str(output_path.parent),
            "--out", output_path.name,
            "--continue=true",
            "--auto-file-renaming=false",
            "--console-log-level=notice",
            "--summary-interval=10",
            # Retry settings for flaky connections
            "--max-tries=5",
            "--retry-wait=3",
            "--timeout=60",
            "--connect-timeout=30",
            url,
        ]

        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("aria2c failed: %s", e)
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return False


def _download_wget(url: str, output_path: Path) -> bool:
    """Download using wget."""
    try:
        cmd = [
            "wget",
            "--continue",
            "--progress=bar:force",
            "-O", str(output_path),
            url,
        ]

        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("wget failed: %s", e)
        return False


def _download_http(url: str, output_path: Path, chunk_size: int = 8 * 1024 * 1024):
    """Download using Python requests with streaming. 8MB chunks for speed."""
    tmp_path = output_path.with_suffix(".part")
    downloaded = 0
    headers = {}
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
    token = _normalize_token(token)
    _enable_hf_transfer()

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
    token = _normalize_token(token)
    _enable_hf_transfer()

    from huggingface_hub import snapshot_download

    logger.info("Ensuring HF subfolder cached: %s/%s", repo_id, subfolder)
    snapshot_download(
        repo_id,
        allow_patterns=[f"{subfolder}/*"],
        token=token,
    )
    return repo_id


def ensure_hf_file_cached(
    repo_id: str,
    filename: str,
    token: str | None = None,
) -> str:
    """
    Download only a specific file from a HuggingFace repo.
    """
    token = _normalize_token(token)
    _enable_hf_transfer()

    from huggingface_hub import hf_hub_download

    logger.info("Ensuring HF file cached: %s/%s", repo_id, filename)
    hf_hub_download(
        repo_id,
        filename,
        token=token,
    )
    return repo_id


# ---------------------------------------------------------------------------
# Parallel Download Orchestrator
# ---------------------------------------------------------------------------

def download_all_models_parallel(
    civitai_model_version_id: str,
    transformer_path: str,
    civitai_token: str | None,
    text_encoder_id: str,
    hf_token: str | None,
    flux2_repo_id: str,
    text_encoder_gguf_file: str | None = None,
    text_encoder_tokenizer_id: str | None = None,
):
    """
    Download ALL model components in parallel for maximum speed.

    Instead of downloading sequentially (transformer → text encoder → VAE),
    this fires off all downloads simultaneously using a thread pool.
    Each individual download already uses its own parallel strategy:
      - CivitAI: aria2c with multi-connection
      - HuggingFace: hf_transfer (Rust-based parallel chunked downloads)

    This function runs them ALL concurrently so total wall time ≈
    max(single download) instead of sum(all downloads).
    """
    # Normalize tokens once at the top level
    civitai_token = _normalize_token(civitai_token)
    hf_token = _normalize_token(hf_token)

    _enable_hf_transfer()

    tasks: dict[str, tuple] = {}

    # Task 1: CivitAI transformer
    tasks["transformer"] = (
        download_civitai_model,
        {"model_version_id": civitai_model_version_id, "output_path": transformer_path, "token": civitai_token},
    )

    # Task 2: Text encoder
    if text_encoder_gguf_file:
        tasks["text_encoder_gguf"] = (
            ensure_hf_file_cached,
            {"repo_id": text_encoder_id, "filename": text_encoder_gguf_file, "token": hf_token},
        )
        tokenizer_id = text_encoder_tokenizer_id or text_encoder_id
        tasks["tokenizer"] = (
            ensure_hf_model_cached,
            {"model_id": tokenizer_id, "token": hf_token},
        )
    else:
        tasks["text_encoder"] = (
            ensure_hf_model_cached,
            {"model_id": text_encoder_id, "token": hf_token},
        )

    # Task 3: VAE
    tasks["vae"] = (
        ensure_hf_subfolder_cached,
        {"repo_id": flux2_repo_id, "subfolder": "vae", "token": hf_token},
    )

    # Task 4: Scheduler
    tasks["scheduler"] = (
        ensure_hf_subfolder_cached,
        {"repo_id": flux2_repo_id, "subfolder": "scheduler", "token": hf_token},
    )

    logger.info(
        "=== Starting PARALLEL download of %d components: %s ===",
        len(tasks), ", ".join(tasks.keys()),
    )
    start_time = time.time()

    results: dict[str, str] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        future_to_name = {
            pool.submit(fn, **kwargs): name
            for name, (fn, kwargs) in tasks.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results[name] = str(result)
                logger.info("✅ %s download complete.", name)
            except Exception as exc:
                errors[name] = str(exc)
                logger.error("❌ %s download FAILED: %s", name, exc)

    elapsed = time.time() - start_time
    logger.info(
        "=== Parallel download finished in %.1fs — %d succeeded, %d failed ===",
        elapsed, len(results), len(errors),
    )

    if errors:
        err_summary = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(f"Some downloads failed: {err_summary}")
