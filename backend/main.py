import io
import inspect
import json
import logging
import textwrap
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel

from backend.settings import AppSettings

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from backend.pipeline_manager import PipelineManager
    PIPELINE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import failure is environment-specific
    PipelineManager = None  # type: ignore[assignment]
    PIPELINE_IMPORT_ERROR = str(exc)


APP_SETTINGS = AppSettings.from_env()
APP_SETTINGS.ensure_directories()
HISTORY_FILE = APP_SETTINGS.output_dir / "history.json"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    input_image_url: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    num_images: int | None = None
    max_sequence_length: int | None = None
    text_encoder_out_layers: str | list[int] | None = None


class LoadLoraRequest(BaseModel):
    name: str
    path: str
    strength: float = 1.0


class UnloadLoraRequest(BaseModel):
    name: str


class LoraStrengthRequest(BaseModel):
    name: str
    strength: float = 1.0


class LoraToggleRequest(BaseModel):
    name: str
    enabled: bool


class PipelineRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.manager = None
        self.state = "idle"
        self.message = "Pipeline has not been initialized yet."
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        manager = self.manager
        return {
            "ready": manager is not None and manager.pipe is not None,
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "loras": manager.get_loaded_loras() if manager else [],
            "resources": manager.describe_resources() if manager else {
                "transformer_path": APP_SETTINGS.transformer_path,
                "transformer_exists": Path(APP_SETTINGS.transformer_path).exists(),
                "transformer_size_bytes": (
                    Path(APP_SETTINGS.transformer_path).stat().st_size
                    if Path(APP_SETTINGS.transformer_path).exists()
                    else 0
                ),
                "text_encoder_id": APP_SETTINGS.text_encoder_id,
                "flux2_repo_id": APP_SETTINGS.flux2_repo_id,
                "local_files_only": not APP_SETTINGS.allow_model_downloads,
            },
        }


runtime = PipelineRuntime()


def _load_history() -> list[dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("History file is unreadable, resetting it.")
    return []


def _save_history(history: list[dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _create_manager() -> Any:
    if PipelineManager is None:
        raise RuntimeError(
            "Pipeline dependencies could not be imported. "
            f"{PIPELINE_IMPORT_ERROR or 'Unknown import failure.'}"
        )
    return PipelineManager(
        transformer_path=APP_SETTINGS.transformer_path,
        text_encoder_id=APP_SETTINGS.text_encoder_id,
        flux2_repo_id=APP_SETTINGS.flux2_repo_id,
        civitai_model_version_id=APP_SETTINGS.civitai_model_version_id,
        hf_token=APP_SETTINGS.hf_token,
        civitai_token=APP_SETTINGS.civitai_token,
        local_files_only=not APP_SETTINGS.allow_model_downloads,
    )


def initialize_pipeline(force: bool = False) -> dict[str, Any]:
    with runtime.lock:
        if runtime.state == "loading":
            return runtime.snapshot()
        if not force and runtime.manager is not None and runtime.manager.pipe is not None:
            return runtime.snapshot()

        runtime.state = "loading"
        runtime.message = "Initializing Flux2 pipeline."
        runtime.error = None
        runtime.started_at = time.time()

        try:
            manager = _create_manager()
            if APP_SETTINGS.allow_model_downloads:
                runtime.message = "Checking model assets and downloading missing files."
                manager.download_models()
            else:
                runtime.message = "Loading only from local files and cache."
            manager.load()
            runtime.manager = manager
            runtime.state = "ready"
            runtime.message = "Pipeline is ready."
            runtime.finished_at = time.time()
            logger.info("Pipeline initialization complete.")
        except Exception as exc:
            runtime.manager = None
            runtime.state = "error"
            runtime.error = str(exc)
            runtime.message = (
                "Pipeline initialization failed. "
                "Check model paths, cached Hugging Face assets, or enable downloads in the runtime environment."
            )
            runtime.finished_at = time.time()
            logger.exception("Pipeline initialization failed")

        return runtime.snapshot()


def _status_payload() -> dict[str, Any]:
    payload = runtime.snapshot()
    payload["settings"] = {
        "auto_initialize": APP_SETTINGS.auto_initialize_pipeline,
        "allow_downloads": APP_SETTINGS.allow_model_downloads,
        "mock_generation": APP_SETTINGS.enable_mock_generation,
    }
    return payload


def _save_generated_images(
    images: list[Image.Image],
    params: dict[str, Any],
    resolved_seed: int,
) -> list[dict[str, Any]]:
    batch_id = uuid.uuid4().hex
    history = _load_history()
    saved_items: list[dict[str, Any]] = []

    for index, image in enumerate(images):
        filename = f"{uuid.uuid4().hex}.png"
        filepath = APP_SETTINGS.output_dir / filename
        image.save(str(filepath), format="PNG")
        item = {
            "id": filename.removesuffix(".png"),
            "batch_id": batch_id,
            "image_index": index,
            "url": f"/outputs/{filename}",
            "prompt": params["prompt"],
            "negative_prompt": params["negative_prompt"],
            "source_image_url": params.get("input_image_url"),
            "width": params["width"],
            "height": params["height"],
            "steps": params["num_inference_steps"],
            "guidance_scale": params["guidance_scale"],
            "seed": resolved_seed,
            "timestamp": time.time(),
        }
        saved_items.append(item)
        history.insert(0, item)

    _save_history(history)
    return saved_items


def _build_mock_images(params: dict[str, Any]) -> dict[str, Any]:
    images: list[Image.Image] = []
    resolved_seed = params["seed"] if params["seed"] >= 0 else int(time.time())
    for index in range(params["num_images"]):
        source_image_url = params.get("input_image_url")
        source_image = _load_input_image_from_url(source_image_url) if source_image_url else None
        image = (
            source_image.convert("RGB").resize((params["width"], params["height"]))
            if source_image is not None
            else Image.new("RGB", (params["width"], params["height"]), color="#f2eee4")
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, params["width"], 180), fill="#13343bcc")
        draw.rectangle((0, params["height"] - 140, params["width"], params["height"]), fill="#c56f4dd9")
        copy = "\n".join(
            textwrap.wrap(
                f"Mock generation {index + 1}\n\nPrompt: {params['prompt']}\n\nNegative: {params['negative_prompt'] or 'none'}",
                width=34,
            )
        )
        draw.text((48, 56), copy, fill="#fff8f0", spacing=8)
        draw.text(
            (48, params["height"] - 108),
            (
                f"{params['width']}x{params['height']} | "
                f"steps {params['num_inference_steps']} | "
                f"guidance {params['guidance_scale']:.1f} | "
                f"seed {resolved_seed}"
            ),
            fill="#221814",
        )
        images.append(image)
    return {"images": images, "seed": resolved_seed}


def _normalize_uploaded_dimension(value: int, key: str) -> int:
    minimum = APP_SETTINGS.generation_limits[key]["min"]
    maximum = APP_SETTINGS.generation_limits[key]["max"]
    step = APP_SETTINGS.generation_limits[key]["step"]
    rounded = int(round(value / step) * step)
    return max(minimum, min(maximum, rounded))


def _resolve_output_path_from_url(url: str | None) -> Path | None:
    if not url:
        return None
    candidate = url.split("?", 1)[0]
    if not candidate.startswith("/outputs/"):
        return None
    output_name = candidate.removeprefix("/outputs/")
    path = (APP_SETTINGS.output_dir / output_name).resolve()
    try:
        path.relative_to(APP_SETTINGS.output_dir.resolve())
    except ValueError:
        return None
    return path if path.exists() else None


def _load_input_image_from_url(url: str | None) -> Image.Image | None:
    path = _resolve_output_path_from_url(url)
    if path is None:
        return None
    with Image.open(path) as image:
        return image.copy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if APP_SETTINGS.auto_initialize_pipeline:
        initialize_pipeline()
    else:
        runtime.state = "idle"
        runtime.message = "Auto-initialization is disabled for this environment."
    yield
    runtime.manager = None
    runtime.state = "idle"
    runtime.message = "Pipeline shut down."


app = FastAPI(title="Flux2 Image Generator", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(APP_SETTINGS.output_dir)), name="outputs")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/config")
async def get_config():
    config = APP_SETTINGS.client_config()
    config["status"] = _status_payload()
    return config


@app.get("/api/status")
async def pipeline_status():
    return _status_payload()


@app.post("/api/pipeline/initialize")
async def initialize_pipeline_endpoint():
    return initialize_pipeline(force=True)


@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    params = APP_SETTINGS.normalize_generation_params(req.model_dump())
    params["input_image_url"] = req.input_image_url
    if not params["prompt"]:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    input_image = _load_input_image_from_url(req.input_image_url)
    if req.input_image_url and input_image is None:
        raise HTTPException(status_code=400, detail="Selected source image could not be found.")

    try:
        if runtime.manager is not None and runtime.manager.pipe is not None:
            allowed_keys = {
                name
                for name in inspect.signature(runtime.manager.generate).parameters
                if name not in {"self", "input_image"}
            }
            generate_params = {
                key: value
                for key, value in params.items()
                if key in allowed_keys
            }
            result = runtime.manager.generate(**generate_params, input_image=input_image)
        elif APP_SETTINGS.enable_mock_generation:
            result = _build_mock_images(params)
        else:
            detail = runtime.error or runtime.message or "Pipeline not ready."
            raise HTTPException(status_code=503, detail=detail)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    saved_items = _save_generated_images(result["images"], params, result["seed"])
    return {"images": saved_items, "count": len(saved_items), "seed": result["seed"]}


@app.get("/api/history")
async def get_history():
    return _load_history()


@app.delete("/api/history/{image_id}")
async def delete_history_item(image_id: str):
    history = _load_history()
    filepath = APP_SETTINGS.output_dir / f"{image_id}.png"
    updated = [item for item in history if item["id"] != image_id]
    _save_history(updated)
    if filepath.exists():
        filepath.unlink()
    return {"status": "ok"}


@app.get("/api/loras")
async def list_loras():
    if runtime.manager is None or runtime.manager.pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    return runtime.manager.get_loaded_loras()


@app.get("/api/loras/available")
async def list_available_loras():
    files = []
    for file_path in sorted(APP_SETTINGS.lora_dir.rglob("*.safetensors")):
        files.append({"name": file_path.stem, "path": str(file_path)})
    return files


@app.post("/api/loras/load")
async def load_lora(req: LoadLoraRequest):
    if runtime.manager is None or runtime.manager.pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        runtime.manager.load_lora(name=req.name, path=req.path, strength=req.strength)
    except Exception as exc:
        logger.exception("LoRA load failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "loras": runtime.manager.get_loaded_loras()}


@app.post("/api/loras/unload")
async def unload_lora(req: UnloadLoraRequest):
    if runtime.manager is None or runtime.manager.pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        runtime.manager.unload_lora(name=req.name)
    except Exception as exc:
        logger.exception("LoRA unload failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "loras": runtime.manager.get_loaded_loras()}


@app.post("/api/loras/strength")
async def set_lora_strength(req: LoraStrengthRequest):
    if runtime.manager is None or runtime.manager.pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        runtime.manager.set_lora_strength(name=req.name, strength=req.strength)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "loras": runtime.manager.get_loaded_loras()}


@app.post("/api/loras/toggle")
async def toggle_lora(req: LoraToggleRequest):
    if runtime.manager is None or runtime.manager.pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        runtime.manager.toggle_lora(name=req.name, enabled=req.enabled)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "loras": runtime.manager.get_loaded_loras()}


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        normalized_width = _normalize_uploaded_dimension(width, "width")
        normalized_height = _normalize_uploaded_dimension(height, "height")

        filename = f"upload_{uuid.uuid4().hex}.png"
        filepath = APP_SETTINGS.output_dir / filename
        image.save(str(filepath), format="PNG")

        return {
            "width": normalized_width,
            "height": normalized_height,
            "original_width": width,
            "original_height": height,
            "url": f"/outputs/{filename}",
        }
    except Exception as exc:
        logger.exception("Image upload failed")
        raise HTTPException(status_code=400, detail=str(exc))
