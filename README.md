# Flux2 Image Generator

Custom GUI application for generating and editing images using [Flux2 Klein 9B](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence) with the `diffusers` library.

## Features
- **Custom Model**: Loads a community-modified Flux2 Klein transformer from a single `.safetensors` file.
- **Custom Text Encoder**: Uses `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (Qwen3-based, abliterated).
- **No Safety Filters**: All safety checkers / watermarkers are disabled.
- **Multi-LoRA**: Load, unload, toggle, and adjust strength of multiple LoRA adapters from the UI in real time.
- **History Tab**: All generated images are saved as lossless PNGs and displayed in a history sidebar.
- **Runtime-driven UI**: Generation defaults, limits, and presets come from the backend instead of being hardcoded in the frontend.
- **Safe Local Dev Mode**: The app can run without downloading models, and can optionally use mock image generation for frontend work.
- **Vanilla Tech Stack**: Pure HTML/CSS/JS frontend — no build step, no framework bloat.

## Quick Start (Lightning.ai Studio)

1. **Copy config**
   ```bash
   cp .env.example .env
   # Edit .env — set your HF_TOKEN, CIVITAI_TOKEN, TRANSFORMER_PATH, etc.
   # In Lightning, set ALLOW_MODEL_DOWNLOADS=true if HF assets are not already cached.
   ```

2. **Download the model**
   Place the downloaded `.safetensors` transformer file from CivitAI into `./models/transformer.safetensors` (or update `TRANSFORMER_PATH` in `.env`).

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run**
   ```bash
   bash run.sh
   ```
   The server starts at `http://0.0.0.0:8080`. Open this in your browser.

5. **Add LoRAs** — Drop `.safetensors` files into `./loras/` and they'll appear in the UI.

## Runtime Modes

- **Lightning runtime**: keep `AUTO_INITIALIZE_PIPELINE=true`. Set `ALLOW_MODEL_DOWNLOADS=true` when the text encoder / VAE / scheduler are not already cached in the environment.
- **Local frontend development**: keep `ALLOW_MODEL_DOWNLOADS=false` and set `ENABLE_MOCK_GENERATION=true` to work on the UI without touching real model assets.

## Flux2 Klein 9B Defaults

- The official `black-forest-labs/FLUX.2-klein-9B` model card describes the 9B model as step-distilled to **4 inference steps**.
- The official Diffusers example for that model uses **`guidance_scale=1.0`** and **`num_inference_steps=4`**.

## Project Structure
```
flux2/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server & endpoints
│   └── pipeline_manager.py  # Diffusers pipeline wrapper
├── frontend/
│   ├── index.html            # Main UI
│   ├── styles.css            # Design system
│   └── app.js                # Client-side logic
├── models/                   # Place transformer .safetensors here
├── loras/                    # Place LoRA .safetensors here
├── outputs/                  # Generated images (auto-created)
├── .env.example              # Configuration template
├── .gitignore
├── requirements.txt
├── run.sh
└── README.md
```
