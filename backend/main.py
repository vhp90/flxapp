import os
import uuid
import time
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import io

from backend.pipeline_manager import PipelineManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
LORA_DIR = Path(os.getenv("LORA_DIR", "./loras"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = OUTPUT_DIR / "history.json"

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------
manager: PipelineManager | None = None


def _load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save_history(history: list[dict]):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    transformer_path = os.getenv("TRANSFORMER_PATH", "./models/transformer.safetensors")
    text_encoder_id = os.getenv("TEXT_ENCODER_ID", "huihui-ai/Huihui-Qwen3-8B-abliterated-v2")
    hf_token = os.getenv("HF_TOKEN")

    manager = PipelineManager(
        transformer_path=transformer_path,
        text_encoder_id=text_encoder_id,
        hf_token=hf_token,
    )
    logger.info("Loading pipeline on startup...")
    manager.load()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Flux2 Image Generator", lifespan=lifespan)

# Serve generated images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1024, ge=256, le=4096)
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=4.0, ge=0.0, le=30.0)
    seed: int = -1
    num_images: int = Field(default=1, ge=1, le=4)


class LoadLoraRequest(BaseModel):
    name: str
    path: str
    strength: float = Field(default=1.0, ge=0.0, le=2.0)


class LoraStrengthRequest(BaseModel):
    name: str
    strength: float = Field(default=1.0, ge=0.0, le=2.0)


class LoraToggleRequest(BaseModel):
    name: str
    enabled: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        images = manager.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            num_images=req.num_images,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    saved_paths = []
    history = _load_history()
    for img in images:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = OUTPUT_DIR / filename
        img.save(str(filepath), format="PNG")  # lossless PNG, no compression
        url = f"/outputs/{filename}"
        saved_paths.append(url)

        history.insert(0, {
            "id": filename.replace(".png", ""),
            "url": url,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "timestamp": time.time(),
        })

    _save_history(history)
    return {"images": saved_paths, "count": len(saved_paths)}


@app.get("/api/history")
async def get_history():
    return _load_history()


@app.delete("/api/history/{image_id}")
async def delete_history_item(image_id: str):
    history = _load_history()
    filepath = OUTPUT_DIR / f"{image_id}.png"
    history = [h for h in history if h["id"] != image_id]
    _save_history(history)
    if filepath.exists():
        filepath.unlink()
    return {"status": "ok"}


# LoRA endpoints
@app.get("/api/loras")
async def list_loras():
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    return manager.get_loaded_loras()


@app.get("/api/loras/available")
async def list_available_loras():
    """List .safetensors files in the loras directory that can be loaded."""
    files = []
    for f in LORA_DIR.glob("*.safetensors"):
        files.append({"name": f.stem, "path": str(f)})
    return files


@app.post("/api/loras/load")
async def load_lora(req: LoadLoraRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        manager.load_lora(name=req.name, path=req.path, strength=req.strength)
    except Exception as e:
        logger.exception("LoRA load failed")
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "loras": manager.get_loaded_loras()}


@app.post("/api/loras/unload")
async def unload_lora(req: LoadLoraRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        manager.unload_lora(name=req.name)
    except Exception as e:
        logger.exception("LoRA unload failed")
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "loras": manager.get_loaded_loras()}


@app.post("/api/loras/strength")
async def set_lora_strength(req: LoraStrengthRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        manager.set_lora_strength(name=req.name, strength=req.strength)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "loras": manager.get_loaded_loras()}


@app.post("/api/loras/toggle")
async def toggle_lora(req: LoraToggleRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        manager.toggle_lora(name=req.name, enabled=req.enabled)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "loras": manager.get_loaded_loras()}


@app.get("/api/status")
async def pipeline_status():
    return {
        "ready": manager is not None and manager.pipe is not None,
        "loras": manager.get_loaded_loras() if manager else [],
    }


# ---------------------------------------------------------------------------
# Quality Presets (up to 4 MP)
# ---------------------------------------------------------------------------
QUALITY_PRESETS = [
    {"name": "0.25 MP", "megapixels": 0.25},
    {"name": "0.5 MP",  "megapixels": 0.5},
    {"name": "1 MP",    "megapixels": 1.0},
    {"name": "1.5 MP",  "megapixels": 1.5},
    {"name": "2 MP",    "megapixels": 2.0},
    {"name": "3 MP",    "megapixels": 3.0},
    {"name": "4 MP",    "megapixels": 4.0},
]


@app.get("/api/presets")
async def get_quality_presets():
    return QUALITY_PRESETS


# ---------------------------------------------------------------------------
# Image Upload — auto detect dimensions
# ---------------------------------------------------------------------------

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Accept an image upload, return its dimensions (auto-sets width/height in UI)."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        w, h = img.size

        # Round to nearest 64 for model compatibility
        w = max(256, min(4096, (w // 64) * 64))
        h = max(256, min(4096, (h // 64) * 64))

        # Save uploaded image to outputs so it can be referenced
        filename = f"upload_{uuid.uuid4().hex}.png"
        filepath = OUTPUT_DIR / filename
        img.save(str(filepath), format="PNG")

        return {
            "width": w,
            "height": h,
            "original_width": img.size[0],
            "original_height": img.size[1],
            "url": f"/outputs/{filename}",
        }
    except Exception as e:
        logger.exception("Image upload failed")
        raise HTTPException(status_code=400, detail=str(e))
