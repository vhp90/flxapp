import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_PRESETS = [
    {"name": "0.5 MP", "megapixels": 0.5},
    {"name": "1 MP", "megapixels": 1.0},
    {"name": "1.5 MP", "megapixels": 1.5},
    {"name": "2 MP", "megapixels": 2.0},
    {"name": "3 MP", "megapixels": 3.0},
    {"name": "4 MP", "megapixels": 4.0},
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_csv_numbers(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values or default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _round_to_multiple(value: int, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


@dataclass(slots=True)
class AppSettings:
    transformer_path: str
    text_encoder_id: str
    flux2_repo_id: str
    civitai_model_version_id: str
    hf_token: str | None
    civitai_token: str | None
    output_dir: Path
    lora_dir: Path
    auto_initialize_pipeline: bool
    allow_model_downloads: bool
    enable_mock_generation: bool
    generation_defaults: dict[str, Any]
    generation_limits: dict[str, dict[str, Any]]
    quality_presets: list[dict[str, float | str]]

    @classmethod
    def from_env(cls) -> "AppSettings":
        quality_values = _env_csv_numbers(
            "QUALITY_PRESET_MEGAPIXELS",
            [preset["megapixels"] for preset in DEFAULT_QUALITY_PRESETS],
        )
        presets = [
            {"name": f"{value:g} MP", "megapixels": float(value)}
            for value in quality_values
        ]
        return cls(
            transformer_path=os.getenv("TRANSFORMER_PATH", "./models/transformer.safetensors"),
            text_encoder_id=os.getenv(
                "TEXT_ENCODER_ID",
                "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
            ),
            flux2_repo_id=os.getenv(
                "FLUX2_REPO_ID",
                "black-forest-labs/FLUX.2-klein-9B",
            ),
            civitai_model_version_id=os.getenv("CIVITAI_MODEL_VERSION_ID", "2746781"),
            hf_token=os.getenv("HF_TOKEN"),
            civitai_token=os.getenv("CIVITAI_TOKEN"),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            lora_dir=Path(os.getenv("LORA_DIR", "./loras")),
            auto_initialize_pipeline=_env_bool("AUTO_INITIALIZE_PIPELINE", True),
            allow_model_downloads=_env_bool("ALLOW_MODEL_DOWNLOADS", False),
            enable_mock_generation=_env_bool("ENABLE_MOCK_GENERATION", False),
            generation_defaults={
                "width": _env_int("DEFAULT_WIDTH", 1024),
                "height": _env_int("DEFAULT_HEIGHT", 1024),
                "num_inference_steps": _env_int("DEFAULT_STEPS", 4),
                "guidance_scale": _env_float("DEFAULT_GUIDANCE_SCALE", 1.0),
                "seed": _env_int("DEFAULT_SEED", -1),
                "num_images": _env_int("DEFAULT_NUM_IMAGES", 1),
            },
            generation_limits={
                "width": {"min": 256, "max": 4096, "step": 64},
                "height": {"min": 256, "max": 4096, "step": 64},
                "num_inference_steps": {"min": 1, "max": 8, "step": 1},
                "guidance_scale": {"min": 0.5, "max": 2.0, "step": 0.1},
                "num_images": {"min": 1, "max": 4, "step": 1},
            },
            quality_presets=presets,
        )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

    def normalize_generation_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.generation_defaults
        limits = self.generation_limits

        def value_or_default(key: str) -> Any:
            value = payload.get(key)
            return defaults[key] if value is None or value == "" else value

        width = self._normalize_dimension(payload.get("width"), "width")
        height = self._normalize_dimension(payload.get("height"), "height")

        steps = int(
            _clamp(
                int(value_or_default("num_inference_steps")),
                limits["num_inference_steps"]["min"],
                limits["num_inference_steps"]["max"],
            )
        )
        guidance = float(
            _clamp(
                float(value_or_default("guidance_scale")),
                limits["guidance_scale"]["min"],
                limits["guidance_scale"]["max"],
            )
        )
        num_images = int(
            _clamp(
                int(value_or_default("num_images")),
                limits["num_images"]["min"],
                limits["num_images"]["max"],
            )
        )
        seed = int(value_or_default("seed"))

        return {
            "prompt": str(payload.get("prompt", "")).strip(),
            "negative_prompt": str(payload.get("negative_prompt", "")).strip(),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "num_images": num_images,
        }

    def client_config(self) -> dict[str, Any]:
        defaults = self.generation_defaults
        limits = self.generation_limits
        return {
            "presets": self.quality_presets,
            "controls": [
                {
                    "key": "num_inference_steps",
                    "label": "Steps",
                    "type": "range",
                    "section": "sampling",
                    "default": defaults["num_inference_steps"],
                    **limits["num_inference_steps"],
                },
                {
                    "key": "guidance_scale",
                    "label": "Guidance scale",
                    "type": "range",
                    "section": "sampling",
                    "default": defaults["guidance_scale"],
                    **limits["guidance_scale"],
                },
                {
                    "key": "seed",
                    "label": "Seed",
                    "type": "number",
                    "section": "sampling",
                    "default": defaults["seed"],
                },
                {
                    "key": "num_images",
                    "label": "Images",
                    "type": "number",
                    "section": "sampling",
                    "default": defaults["num_images"],
                    **limits["num_images"],
                },
            ],
            "dimensions": {
                "width": {
                    "default": defaults["width"],
                    **limits["width"],
                },
                "height": {
                    "default": defaults["height"],
                    **limits["height"],
                },
            },
            "pipeline": {
                "auto_initialize": self.auto_initialize_pipeline,
                "allow_downloads": self.allow_model_downloads,
                "mock_generation": self.enable_mock_generation,
            },
            "model": {
                "transformer_path": self.transformer_path,
                "text_encoder_id": self.text_encoder_id,
                "flux2_repo_id": self.flux2_repo_id,
                "civitai_model_version_id": self.civitai_model_version_id,
            },
        }

    def _normalize_dimension(self, raw_value: Any, key: str) -> int:
        default = self.generation_defaults[key]
        value = int(raw_value if raw_value is not None else default)
        minimum = self.generation_limits[key]["min"]
        maximum = self.generation_limits[key]["max"]
        step = self.generation_limits[key]["step"]
        rounded = _round_to_multiple(value, step)
        return int(_clamp(rounded, minimum, maximum))
