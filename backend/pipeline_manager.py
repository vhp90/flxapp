"""
Pipeline manager for Flux2KleinPipeline.

Handles:
  - Auto-downloading models (CivitAI transformer, HuggingFace text encoder + VAE)
  - Assembling the pipeline with custom components
  - Removing all safety filters
  - Dynamic multi-LoRA management (load / unload / toggle / strength)
"""

import os
import json
import random
import torch
import logging
from functools import wraps
from pathlib import Path
from PIL import Image
from safetensors import safe_open

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
    ensure_hf_file_cached,
    download_all_models_parallel,
)

logger = logging.getLogger(__name__)


def _patch_diffusers_flux2_key_map():
    """
    Monkey-patch diffusers' FLUX2 single-file converter for older releases.

    Some diffusers builds ship `convert_flux2_transformer_checkpoint_to_diffusers`
    without stripping the `model.diffusion_model.` prefix that appears in
    CivitAI/BFL checkpoints. That makes `from_single_file()` look up keys like
    `double_blocks.0.img_attn.norm.key_norm` in the wrong map and crash with a
    `KeyError`.

    We normalize the checkpoint keys before delegating to diffusers' native
    converter, then update every reference used by `from_single_file()` so the
    patched function is guaranteed to run.
    """
    try:
        import diffusers.loaders.single_file_model as sfm
        import diffusers.loaders.single_file_utils as sfu

        original_converter = getattr(sfu, "convert_flux2_transformer_checkpoint_to_diffusers")
        if getattr(original_converter, "_flxapp_flux2_patch", False):
            logger.info("FLUX2 single-file converter already patched.")
            return

        @wraps(original_converter)
        def patched_converter(checkpoint, **kwargs):
            normalized_checkpoint = {
                key.removeprefix("model.diffusion_model."): value
                for key, value in checkpoint.items()
            }
            return original_converter(dict(normalized_checkpoint), **kwargs)

        patched_converter._flxapp_flux2_patch = True

        sfu.convert_flux2_transformer_checkpoint_to_diffusers = patched_converter
        sfm.convert_flux2_transformer_checkpoint_to_diffusers = patched_converter

        loadable_class = sfm.SINGLE_FILE_LOADABLE_CLASSES.get("Flux2Transformer2DModel")
        if loadable_class is not None:
            loadable_class["checkpoint_mapping_fn"] = patched_converter

        logger.info(
            "Patched diffusers FLUX2 single-file converter to normalize "
            "`model.diffusion_model.` checkpoint prefixes."
        )

    except (AttributeError, ImportError) as e:
        logger.warning("Could not patch diffusers FLUX2 single-file converter: %s", e)


def _infer_flux2_transformer_config(transformer_path: str) -> tuple[dict[str, object], Path]:
    """
    Build a local diffusers config that matches the downloaded transformer checkpoint.

    This avoids diffusers' checkpoint heuristics selecting the wrong upstream repo
    (for example `FLUX.2-dev`) when loading a community `.safetensors` file.
    """
    checkpoint_path = Path(transformer_path)
    config_root = checkpoint_path.parent / f"{checkpoint_path.stem}.diffusers_config"
    config_dir = config_root / "transformer"
    config_path = config_dir / "config.json"

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

        def get_shape(key: str) -> tuple[int, ...]:
            return tuple(handle.get_tensor(key).shape)

        img_in_shape = get_shape("model.diffusion_model.img_in.weight")
        txt_in_shape = get_shape("model.diffusion_model.txt_in.weight")
        time_in_shape = get_shape("model.diffusion_model.time_in.in_layer.weight")
        final_out_shape = get_shape("model.diffusion_model.final_layer.linear.weight")
        try:
            q_norm_shape = get_shape("model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight")
            attention_head_dim = q_norm_shape[0]
        except Exception:
            attention_head_dim = 128  # Default for FLUX architecture if QK norm is absent
            
        double_mlp_in_shape = get_shape("model.diffusion_model.double_blocks.0.img_mlp.0.weight")

        inner_dim, in_channels = img_in_shape
        out_channels = final_out_shape[0]
        num_attention_heads = inner_dim // attention_head_dim
        joint_attention_dim = txt_in_shape[1]
        timestep_guidance_channels = time_in_shape[1]
        num_layers = len(
            {
                int(key.split(".")[3])
                for key in keys
                if key.startswith("model.diffusion_model.double_blocks.")
            }
        )
        num_single_layers = len(
            {
                int(key.split(".")[3])
                for key in keys
                if key.startswith("model.diffusion_model.single_blocks.")
            }
        )
        guidance_embeds = "model.diffusion_model.guidance_in.in_layer.weight" in keys
        mlp_ratio = double_mlp_in_shape[0] / (2 * inner_dim)

    config = {
        "_class_name": "Flux2Transformer2DModel",
        "_diffusers_version": "0.37.0.dev0",
        "patch_size": 1,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "attention_head_dim": attention_head_dim,
        "num_attention_heads": num_attention_heads,
        "joint_attention_dim": joint_attention_dim,
        "timestep_guidance_channels": timestep_guidance_channels,
        "mlp_ratio": mlp_ratio,
        "axes_dims_rope": [32, 32, 32, 32],
        "rope_theta": 2000,
        "eps": 1e-6,
        "guidance_embeds": guidance_embeds,
    }

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return config, config_root


class PipelineManager:
    """
    Manages a Flux2KleinPipeline instance with dynamic LoRA loading/unloading.
    Auto-downloads all required models on first run.
    """

    def __init__(
        self,
        transformer_path: str,
        transformer_dtype: str,
        text_encoder_id: str,
        flux2_repo_id: str,
        civitai_model_version_id: str,
        hf_token: str | None = None,
        civitai_token: str | None = None,
        local_files_only: bool = False,
        text_encoder_gguf_file: str | None = None,
        text_encoder_tokenizer_id: str | None = None,
        low_vram_mode: bool = False,
    ):
        self.transformer_path = transformer_path
        self.transformer_dtype = transformer_dtype
        self.text_encoder_id = text_encoder_id
        self.flux2_repo_id = flux2_repo_id
        self.civitai_model_version_id = civitai_model_version_id
        self.hf_token = hf_token
        self.civitai_token = civitai_token
        self.local_files_only = local_files_only
        self.text_encoder_gguf_file = text_encoder_gguf_file
        self.text_encoder_tokenizer_id = text_encoder_tokenizer_id
        self.low_vram_mode = low_vram_mode
        self.pipe = None
        self.loaded_loras: dict[str, dict] = {}
        self._adapter_counter = 0

    # ------------------------------------------------------------------
    # Download & Load
    # ------------------------------------------------------------------

    def download_models(self):
        """
        Download all required models in parallel for maximum speed.

        Fires off CivitAI transformer + HF text encoder + HF VAE/scheduler
        downloads simultaneously. Each individual download also uses its own
        parallelism (aria2c 16-conn for CivitAI, hf_transfer for HF).
        """
        download_all_models_parallel(
            civitai_model_version_id=self.civitai_model_version_id,
            transformer_path=self.transformer_path,
            civitai_token=self.civitai_token,
            text_encoder_id=self.text_encoder_id,
            hf_token=self.hf_token,
            flux2_repo_id=self.flux2_repo_id,
            text_encoder_gguf_file=self.text_encoder_gguf_file,
            text_encoder_tokenizer_id=self.text_encoder_tokenizer_id,
        )

    def load(self):
        """Load the full pipeline onto GPU from downloaded components."""

        # Patch any missing key mappings in diffusers before loading
        _patch_diffusers_flux2_key_map()

        transformer_config, transformer_config_root = _infer_flux2_transformer_config(self.transformer_path)
        logger.info(
            "Using local transformer config inferred from checkpoint: %s "
            "(dim=%s, double_layers=%s, single_layers=%s, heads=%s x %s)",
            transformer_config_root / "transformer" / "config.json",
            transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"],
            transformer_config["num_layers"],
            transformer_config["num_single_layers"],
            transformer_config["num_attention_heads"],
            transformer_config["attention_head_dim"],
        )

        # ---------- Transformer (from CivitAI safetensors) ----------
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp8": torch.float8_e4m3fn,
        }
        torch_dtype = dtype_map.get(self.transformer_dtype.lower(), None)
        logger.info("Loading transformer from: %s (dtype: %s, low_vram: %s)", self.transformer_path, torch_dtype, self.low_vram_mode)

        if self.low_vram_mode:
            # In low VRAM mode, load transformer directly to CPU first,
            # the pipeline's cpu_offload will move it on-demand.
            transformer = Flux2Transformer2DModel.from_single_file(
                self.transformer_path,
                config=str(transformer_config_root),
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )
        else:
            transformer = Flux2Transformer2DModel.from_single_file(
                self.transformer_path,
                config=str(transformer_config_root),
                subfolder="transformer",
                torch_dtype=torch_dtype,
            ).to("cuda")

        # ---------- Text Encoder + Tokenizer ----------
        tokenizer_id = self.text_encoder_tokenizer_id or self.text_encoder_id
        logger.info("Loading tokenizer: %s", tokenizer_id)
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            tokenizer_id,
            token=self.hf_token,
            local_files_only=self.local_files_only,
        )
        
        if self.text_encoder_gguf_file:
            logger.info("Loading GGUF text encoder: %s from %s", self.text_encoder_gguf_file, self.text_encoder_id)
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.text_encoder_id,
                gguf_file=self.text_encoder_gguf_file,
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )
        else:
            logger.info("Loading text encoder: %s", self.text_encoder_id)
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.text_encoder_id,
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )

        if not self.low_vram_mode:
            text_encoder = text_encoder.to("cuda")

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
            )
        except (ImportError, Exception) as e:
            logger.warning("AutoencoderKLFlux2 not available (%s), trying AutoencoderKL...", e)
            vae = AutoencoderKL.from_pretrained(
                self.flux2_repo_id,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                local_files_only=self.local_files_only,
            )

        if not self.low_vram_mode:
            vae = vae.to("cuda")

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

        if self.low_vram_mode:
            # ----------------------------------------------------------
            # Smart Group Offloading (much faster than sequential offload)
            #
            # Instead of moving individual layers one-by-one, we use
            # diffusers' group offloading with CUDA streams to:
            # 1. Group layers together into batches
            # 2. Overlap GPU compute with CPU↔GPU data transfer
            # 3. Keep peak VRAM under control (~8-9 GB on T4)
            #
            # Strategy:
            # - Transformer: leaf_level offload with CUDA stream
            #   (runs many denoising steps, needs to be fast)
            # - Text encoder: block_level offload, groups of 4 layers
            #   (runs once per generation, ~2GB per group)
            # - VAE: leaf_level offload with CUDA stream
            #   (runs once for decode, small model)
            # ----------------------------------------------------------
            try:
                from diffusers.hooks import apply_group_offloading

                onload_device = torch.device("cuda")
                offload_device = torch.device("cpu")

                # Transformer: leaf-level + CUDA stream for max denoising speed
                # Each leaf module loads to GPU right before compute, async prefetch
                apply_group_offloading(
                    self.pipe.transformer,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                    use_stream=True,
                    non_blocking=True,
                )
                logger.info("Transformer: leaf-level group offload with CUDA stream enabled.")

                # Text encoder: block-level, groups of 4 transformer layers
                # ~2GB per group on GPU, fast enough since it only runs once
                apply_group_offloading(
                    self.pipe.text_encoder,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="block_level",
                    num_blocks_per_group=4,
                )
                logger.info("Text encoder: block-level group offload (4 layers/group) enabled.")

                # VAE: leaf-level + CUDA stream
                apply_group_offloading(
                    self.pipe.vae,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type="leaf_level",
                    use_stream=True,
                    non_blocking=True,
                )
                logger.info("VAE: leaf-level group offload with CUDA stream enabled.")

                logger.info(
                    "LOW VRAM MODE: smart group offloading active — "
                    "peak VRAM ~8-9 GB, overlapping compute with data transfer."
                )

            except (ImportError, AttributeError) as e:
                # Fallback to sequential offload if group offloading isn't available
                logger.warning(
                    "Group offloading not available (%s), falling back to sequential CPU offload.", e
                )
                self.pipe.enable_sequential_cpu_offload()
                logger.info("LOW VRAM MODE: sequential CPU offload enabled (fallback).")
        else:
            # Move entire pipeline to GPU
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
            "transformer_dtype": self.transformer_dtype,
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
