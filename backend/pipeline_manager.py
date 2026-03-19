"""
Pipeline manager for Flux2KleinPipeline.

Handles:
  - Auto-downloading models (CivitAI transformer, HuggingFace text encoder + VAE)
  - Assembling the pipeline with custom components
  - Removing all safety filters
  - Dynamic multi-LoRA management (load / unload / toggle / strength)
"""

import os
import random
import torch
import logging
from pathlib import Path
from PIL import Image

from diffusers import (
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
)
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

from backend.model_downloader import (
    download_civitai_model,
    ensure_hf_model_cached,
    ensure_hf_subfolder_cached,
)

logger = logging.getLogger(__name__)


def _patch_diffusers_flux2_key_map():
    """
    Monkey-patch missing BFL native-format key mappings into diffusers'
    FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP.

    The CivitAI safetensors uses BFL/ComfyUI native key names such as
    `img_attn.norm.key_norm` and `img_attn.norm.query_norm` which were
    not included in older diffusers releases. This patch adds them so
    from_single_file works regardless of installed diffusers version.
    """
    try:
        import diffusers.loaders.single_file_utils as sfu

        # Keys present in BFL/ComfyUI native checkpoints that older
        # diffusers builds don't know how to remap.
        missing = {
            # img (image) stream norms
            "img_attn.norm.key_norm":   "attn.norm_k",
            "img_attn.norm.query_norm": "attn.norm_q",
            # txt (text) stream norms
            "txt_attn.norm.key_norm":   "attn.norm_k",
            "txt_attn.norm.query_norm": "attn.norm_q",
        }

        patched = 0
        for native_key, diffusers_key in missing.items():
            if native_key not in sfu.FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP:
                sfu.FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP[native_key] = diffusers_key
                patched += 1

        if patched:
            logger.info(
                "Patched %d missing key(s) into FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP "
                "(diffusers version too old).",
                patched,
            )
        else:
            logger.info("FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP already up to date.")

    except (AttributeError, ImportError) as e:
        logger.warning("Could not patch FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP: %s", e)


class PipelineManager:
    """
    Manages a Flux2KleinPipeline instance with dynamic LoRA loading/unloading.
    Auto-downloads all required models on first run.
    """

    def __init__(
        self,
        transformer_path: str,
        text_encoder_id: str,
        flux2_repo_id: str,
        civitai_model_version_id: str,
        hf_token: str | None = None,
        civitai_token: str | None = None,
        local_files_only: bool = False,
    ):
        self.transformer_path = transformer_path
        self.text_encoder_id = text_encoder_id
        self.flux2_repo_id = flux2_repo_id
        self.civitai_model_version_id = civitai_model_version_id
        self.hf_token = hf_token
        self.civitai_token = civitai_token
        self.local_files_only = local_files_only
        self.pipe = None
        self.loaded_loras: dict[str, dict] = {}
        self._adapter_counter = 0

    # ------------------------------------------------------------------
    # Download & Load
    # ------------------------------------------------------------------

    def download_models(self):
        """Download all required models if not already present."""

        # 1. CivitAI transformer
        logger.info("=== Step 1/3: Checking CivitAI transformer ===")
        download_civitai_model(
            model_version_id=self.civitai_model_version_id,
            output_path=self.transformer_path,
            token=self.civitai_token,
        )

        # 2. HuggingFace text encoder + tokenizer
        logger.info("=== Step 2/3: Checking text encoder (%s) ===", self.text_encoder_id)
        ensure_hf_model_cached(self.text_encoder_id, token=self.hf_token)

        # 3. HuggingFace VAE + scheduler configs from official Flux2 repo
        logger.info("=== Step 3/3: Checking VAE & scheduler from %s ===", self.flux2_repo_id)
        ensure_hf_subfolder_cached(self.flux2_repo_id, "vae", token=self.hf_token)
        ensure_hf_subfolder_cached(self.flux2_repo_id, "scheduler", token=self.hf_token)

        logger.info("=== All models ready ===")

    def load(self):
        """Load the full pipeline onto GPU from downloaded components."""

        # Patch any missing key mappings in diffusers before loading
        _patch_diffusers_flux2_key_map()

        # ---------- Transformer (from CivitAI safetensors) ----------
        logger.info("Loading transformer from: %s", self.transformer_path)
        transformer = Flux2Transformer2DModel.from_single_file(
            self.transformer_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # ---------- Text Encoder + Tokenizer ----------
        logger.info("Loading text encoder: %s", self.text_encoder_id)
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            self.text_encoder_id,
            token=self.hf_token,
            local_files_only=self.local_files_only,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self.text_encoder_id,
            torch_dtype=torch.bfloat16,
            token=self.hf_token,
            local_files_only=self.local_files_only,
        ).to("cuda")

        # ---------- VAE (from official Flux2 repo) ----------
        logger.info("Loading VAE from: %s", self.flux2_repo_id)
        try:
            # Try Flux2-specific VAE class if available
            from diffusers import AutoencoderKLFlux2
            vae = AutoencoderKLFlux2.from_pretrained(
                self.flux2_repo_id,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            ).to("cuda")
        except (ImportError, Exception) as e:
            logger.warning("AutoencoderKLFlux2 not available (%s), trying AutoencoderKL...", e)
            vae = AutoencoderKL.from_pretrained(
                self.flux2_repo_id,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            ).to("cuda")

        # ---------- Scheduler ----------
        logger.info("Loading scheduler from: %s", self.flux2_repo_id)
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.flux2_repo_id,
                subfolder="scheduler",
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )
        except Exception:
            logger.warning("Could not load scheduler config from repo, using defaults.")
            scheduler = FlowMatchEulerDiscreteScheduler()

        # ---------- Assemble Pipeline ----------
        logger.info("Assembling Flux2KleinPipeline...")
        self.pipe = Flux2KleinPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae,
        )

        # ---------- Remove ALL safety filters ----------
        for attr in [
            "safety_checker", "feature_extractor",
            "watermarker", "nsfw_checker", "content_filter",
        ]:
            if hasattr(self.pipe, attr):
                setattr(self.pipe, attr, None)
                logger.info("Disabled safety component: %s", attr)

        # Also patch any _check methods that might block content
        if hasattr(self.pipe, "run_safety_checker"):
            self.pipe.run_safety_checker = lambda *a, **kw: (a[0] if a else None, None)
            logger.info("Patched run_safety_checker.")

        # Move entire pipeline to GPU — 96 GB VRAM available
        self.pipe.to("cuda")
        logger.info("Pipeline loaded on GPU successfully.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int = -1,
        num_images: int = 1,
        text_encoder_out_layers: tuple = (9, 18, 27),
        input_image: Image.Image | None = None,
    ):
        """Generate images with current pipeline state."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        self._apply_active_loras()

        resolved_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cuda").manual_seed(resolved_seed)

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if input_image is not None:
            kwargs["image"] = input_image

        if negative_prompt:
            kwargs["negative_prompt_embeds"] = negative_prompt

        result = self.pipe(**kwargs)
        return {
            "images": result.images,
            "seed": resolved_seed,
        }

    # ------------------------------------------------------------------
    # LoRA Management
    # ------------------------------------------------------------------

    def load_lora(self, name: str, path: str, strength: float = 1.0):
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded.")
        if name in self.loaded_loras:
            raise ValueError(f"LoRA '{name}' is already loaded. Unload it first.")

        adapter_name = f"lora_{self._adapter_counter}"
        self._adapter_counter += 1

        logger.info("Loading LoRA '%s' from %s (adapter=%s)", name, path, adapter_name)
        self.pipe.load_lora_weights(path, adapter_name=adapter_name)

        self.loaded_loras[name] = {
            "path": path,
            "strength": strength,
            "adapter_name": adapter_name,
            "enabled": True,
        }
        logger.info("LoRA '%s' loaded with strength %.2f", name, strength)

    def unload_lora(self, name: str):
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA '{name}' is not loaded.")
        adapter_name = self.loaded_loras[name]["adapter_name"]
        logger.info("Unloading LoRA '%s' (adapter=%s)", name, adapter_name)
        self.pipe.delete_adapters([adapter_name])
        del self.loaded_loras[name]

    def set_lora_strength(self, name: str, strength: float):
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA '{name}' is not loaded.")
        self.loaded_loras[name]["strength"] = strength
        logger.info("LoRA '%s' strength set to %.2f", name, strength)

    def toggle_lora(self, name: str, enabled: bool):
        if name not in self.loaded_loras:
            raise ValueError(f"LoRA '{name}' is not loaded.")
        self.loaded_loras[name]["enabled"] = enabled
        logger.info("LoRA '%s' %s", name, "enabled" if enabled else "disabled")

    def get_loaded_loras(self) -> list[dict]:
        return [
            {"name": n, "path": i["path"], "strength": i["strength"], "enabled": i["enabled"]}
            for n, i in self.loaded_loras.items()
        ]

    def describe_resources(self) -> dict:
        transformer = Path(self.transformer_path)
        return {
            "transformer_path": self.transformer_path,
            "transformer_exists": transformer.exists(),
            "transformer_size_bytes": transformer.stat().st_size if transformer.exists() else 0,
            "text_encoder_id": self.text_encoder_id,
            "flux2_repo_id": self.flux2_repo_id,
            "local_files_only": self.local_files_only,
        }

    def _apply_active_loras(self):
        active_adapters = []
        active_weights = []
        for info in self.loaded_loras.values():
            if info["enabled"]:
                active_adapters.append(info["adapter_name"])
                active_weights.append(info["strength"])

        if active_adapters:
            self.pipe.set_adapters(active_adapters, adapter_weights=active_weights)
        else:
            try:
                self.pipe.set_adapters([], adapter_weights=[])
            except Exception:
                pass
