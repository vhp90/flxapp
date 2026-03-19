const $ = (selector) => document.querySelector(selector);

const DOM = {
  status: $("#pipeline-status"),
  statusBanner: $("#status-banner"),
  initPipelineBtn: $("#init-pipeline-btn"),
  prompt: $("#prompt-input"),
  negativePrompt: $("#negative-prompt-input"),
  generateBtn: $("#generate-btn"),
  generateSpinner: $("#generate-spinner"),
  clearResultsBtn: $("#clear-results-btn"),
  imageUpload: $("#image-upload"),
  clearSourceBtn: $("#clear-source-btn"),
  width: $("#width-input"),
  height: $("#height-input"),
  megapixelDisplay: $("#megapixel-display"),
  aspectDisplay: $("#aspect-display"),
  presetSelect: $("#preset-select"),
  outputWidth: $("#output-width"),
  outputHeight: $("#output-height"),
  outputMp: $("#output-mp"),
  samplingControls: $("#sampling-controls"),
  loraList: $("#lora-list"),
  loraNameInput: $("#lora-name-input"),
  loraPathInput: $("#lora-path-input"),
  loraStrengthInput: $("#lora-strength-input"),
  loraStrengthValue: $("#lora-strength-value"),
  loraAddBtn: $("#lora-add-btn"),
  availableLorasList: $("#available-loras-list"),
  runtimeFacts: $("#runtime-facts"),
  sourceCard: $("#source-card"),
  sourceTitle: $("#source-title"),
  sourceDescription: $("#source-description"),
  stageTitle: $("#stage-title"),
  canvasMeta: $("#canvas-meta"),
  stagePlaceholder: $("#stage-placeholder"),
  singleStage: $("#single-stage"),
  singleStageImage: $("#single-stage-image"),
  singleStageLabel: $("#single-stage-label"),
  compareStage: $("#compare-stage"),
  compareBeforeImage: $("#compare-before-image"),
  compareAfterImage: $("#compare-after-image"),
  compareAfterShell: $("#compare-after-shell"),
  compareDivider: $("#compare-divider"),
  compareSlider: $("#compare-slider"),
  compareSurface: $("#compare-surface"),
  compareModeBadge: $("#compare-mode-badge"),
  fullscreenStageBtn: $("#fullscreen-stage-btn"),
  useActiveAsSourceBtn: $("#use-active-as-source-btn"),
  resultStrip: $("#result-strip"),
  resultCount: $("#result-count"),
  historyList: $("#history-list"),
  stagePanel: $(".stage-panel"),
};

const state = {
  config: null,
  status: null,
  history: [],
  results: [],
  controlInputs: {},
  sourceImage: null,
  activeResult: null,
  isGenerating: false,
  comparePosition: 50,
  isDraggingCompare: false,
};

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Request failed.");
  }
  return response.json();
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

function escapeAttr(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function truncate(value, maxLength) {
  return value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;
}

function gcd(a, b) {
  return b === 0 ? a : gcd(b, a % b);
}

function formatCount(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function updateBanner(message, tone = "") {
  if (!message) {
    DOM.statusBanner.textContent = "";
    DOM.statusBanner.className = "status-banner hidden";
    return;
  }

  DOM.statusBanner.textContent = message;
  DOM.statusBanner.className = `status-banner ${tone}`.trim();
}

function renderStatus(status) {
  state.status = status;

  let label = "Runtime idle";
  let tone = "offline";
  if (status.ready) {
    label = "Pipeline ready";
    tone = "online";
  } else if (status.state === "loading") {
    label = "Initializing";
    tone = "offline";
  } else if (status.state === "error") {
    label = "Needs attention";
    tone = "error";
  }

  DOM.status.textContent = label;
  DOM.status.className = `status-pill ${tone}`;

  const settings = status.settings || {};
  const extraFlags = [];
  if (settings.mock_generation) extraFlags.push("mock generation enabled");
  if (!settings.allow_downloads) extraFlags.push("local files only");

  if (status.state === "error") {
    updateBanner(`${status.message}${status.error ? ` ${status.error}` : ""}`, "error");
  } else if (!status.ready) {
    updateBanner(
      `${status.message || "Pipeline is not ready yet."}${extraFlags.length ? ` (${extraFlags.join(", ")})` : ""}`,
      "warning",
    );
  } else {
    updateBanner(`Runtime ready${extraFlags.length ? ` (${extraFlags.join(", ")})` : ""}.`);
  }

  DOM.initPipelineBtn.classList.toggle("hidden", status.ready);
  syncGenerateButton();
  renderRuntimeFacts();
}

function renderRuntimeFacts() {
  if (!state.config || !state.status) return;

  const model = state.config.model || {};
  const resources = state.status.resources || {};
  const settings = state.status.settings || {};
  const facts = [
    ["Transformer file", model.transformer_path || "Not configured"],
    ["Transformer present", resources.transformer_exists ? "Yes" : "No"],
    ["Text encoder", model.text_encoder_id || "Not configured"],
    ["Flux2 repo", model.flux2_repo_id || "Not configured"],
    ["CivitAI version", model.civitai_model_version_id || "Not configured"],
    ["Downloads allowed", settings.allow_downloads ? "Yes" : "No"],
    ["Mock generation", settings.mock_generation ? "Yes" : "No"],
  ];

  DOM.runtimeFacts.innerHTML = facts
    .map(
      ([label, value]) => `
        <div>
          <dt>${escapeHtml(label)}</dt>
          <dd>${escapeHtml(String(value))}</dd>
        </div>
      `,
    )
    .join("");
}

function buildControl(control) {
  const wrapper = document.createElement("div");
  wrapper.className = "control-card";

  const valueMarkup = control.type === "range"
    ? `<span class="slider-value" data-value-for="${control.key}"></span>`
    : "";

  wrapper.innerHTML = `
    <div class="control-topline">
      <strong>${escapeHtml(control.label)}</strong>
      ${valueMarkup}
    </div>
    <input
      id="control-${control.key}"
      type="${control.type === "range" ? "range" : control.type}"
      ${control.min !== undefined ? `min="${control.min}"` : ""}
      ${control.max !== undefined ? `max="${control.max}"` : ""}
      ${control.step !== undefined ? `step="${control.step}"` : ""}
      ${control.placeholder ? `placeholder="${escapeAttr(control.placeholder)}"` : ""}
      value="${escapeAttr(String(control.default ?? ""))}"
    />
  `;

  const input = wrapper.querySelector("input");
  state.controlInputs[control.key] = input;

  if (control.type === "range") {
    const valueNode = wrapper.querySelector(`[data-value-for="${control.key}"]`);
    const updateValue = () => {
      valueNode.textContent = Number(input.value).toFixed(control.step < 1 ? 1 : 0);
    };
    updateValue();
    input.addEventListener("input", updateValue);
  }

  return wrapper;
}

function renderControls() {
  state.controlInputs = {};
  DOM.samplingControls.innerHTML = "";

  for (const control of state.config?.controls || []) {
    DOM.samplingControls.appendChild(buildControl(control));
  }
}

function renderPresets() {
  DOM.presetSelect.innerHTML = '<option value="native">Native dimensions</option>';
  for (const preset of state.config?.presets || []) {
    const option = document.createElement("option");
    option.value = String(preset.megapixels);
    option.textContent = `${preset.name} target`;
    DOM.presetSelect.appendChild(option);
  }
}

function applyConfigDefaults() {
  const dimensions = state.config?.dimensions;
  if (!dimensions) return;

  DOM.width.value = dimensions.width.default;
  DOM.height.value = dimensions.height.default;
  DOM.width.min = dimensions.width.min;
  DOM.width.max = dimensions.width.max;
  DOM.width.step = dimensions.width.step;
  DOM.height.min = dimensions.height.min;
  DOM.height.max = dimensions.height.max;
  DOM.height.step = dimensions.height.step;
  DOM.presetSelect.value = "native";

  updateResolutionMeta();
}

function calcOutputDims(baseWidth, baseHeight, targetMP) {
  const step = Number(state.config?.dimensions?.width?.step || 64);
  const min = Number(state.config?.dimensions?.width?.min || 256);
  const max = Number(state.config?.dimensions?.width?.max || 4096);
  const aspect = baseWidth / baseHeight;
  const rawHeight = Math.sqrt((targetMP * 1000000) / aspect);
  const rawWidth = rawHeight * aspect;
  const normalize = (value) => Math.max(min, Math.min(max, Math.round(value / step) * step));
  return { width: normalize(rawWidth), height: normalize(rawHeight) };
}

function getOutputDims() {
  const baseWidth = Number(DOM.width.value || 1024);
  const baseHeight = Number(DOM.height.value || 1024);
  if (DOM.presetSelect.value === "native") {
    return { width: baseWidth, height: baseHeight };
  }
  return calcOutputDims(baseWidth, baseHeight, Number(DOM.presetSelect.value));
}

function updateResolutionMeta() {
  const width = Number(DOM.width.value || 1024);
  const height = Number(DOM.height.value || 1024);
  const ratio = gcd(width, height);
  const mp = (width * height) / 1000000;
  const output = getOutputDims();

  DOM.megapixelDisplay.textContent = `${mp.toFixed(2)} MP`;
  DOM.aspectDisplay.textContent = `${width / ratio}:${height / ratio}`;
  DOM.outputWidth.textContent = output.width;
  DOM.outputHeight.textContent = output.height;
  DOM.outputMp.textContent = `${((output.width * output.height) / 1000000).toFixed(2)} MP`;
}

function readControlValue(key) {
  return state.controlInputs[key]?.value ?? "";
}

function generationPayload() {
  const output = getOutputDims();
  return {
    prompt: DOM.prompt.value.trim(),
    negative_prompt: DOM.negativePrompt.value.trim(),
    input_image_url: state.sourceImage?.url || null,
    width: output.width,
    height: output.height,
    num_inference_steps: Number(readControlValue("num_inference_steps")),
    guidance_scale: Number(readControlValue("guidance_scale")),
    seed: Number(readControlValue("seed")),
    num_images: Number(readControlValue("num_images")),
  };
}

function syncGenerateButton() {
  DOM.generateBtn.disabled = state.isGenerating || !((state.status?.ready || state.status?.settings?.mock_generation) ?? false);
}

function setGenerating(isGenerating) {
  state.isGenerating = isGenerating;
  DOM.generateSpinner.classList.toggle("hidden", !isGenerating);
  DOM.generateBtn.querySelector(".btn-label").textContent = isGenerating ? "Generating…" : "Generate";
  syncGenerateButton();
}

function setSourceImage(image) {
  state.sourceImage = image;
  renderSourceCard();
  renderStage();
}

function renderSourceCard() {
  if (!state.sourceImage) {
    DOM.sourceCard.className = "source-card empty";
    DOM.sourceCard.innerHTML = `
      <div class="source-preview empty-state">
        <p>No source image</p>
        <span>Upload one to anchor dimensions.</span>
      </div>
      <div class="source-copy">
        <strong id="source-title">No source selected</strong>
        <span id="source-description">Upload an image to start.</span>
      </div>
    `;
    return;
  }

  DOM.sourceCard.className = "source-card";
  DOM.sourceCard.innerHTML = `
    <div class="source-preview">
      <img src="${state.sourceImage.url}" alt="Current source image" />
    </div>
    <div class="source-copy">
      <strong>${escapeHtml(state.sourceImage.label)}</strong>
      <span>${escapeHtml(state.sourceImage.description)}</span>
      <div class="stage-label-row">
        <span class="source-tag">${state.sourceImage.width}×${state.sourceImage.height}</span>
        <span class="source-tag">${escapeHtml(state.sourceImage.kind)}</span>
      </div>
    </div>
  `;
}

/* ================================================================
   Compare slider — interactive mouse/touch drag on the image
   ================================================================ */
function updateCompareSurface() {
  const position = `${state.comparePosition}%`;
  DOM.compareStage.style.setProperty("--compare-position", position);
  DOM.compareDivider.style.left = position;
  // keep the hidden range in sync (for any external reads)
  DOM.compareSlider.value = state.comparePosition;
}

function getComparePositionFromEvent(event) {
  if (!DOM.compareSurface) return 50;
  const rect = DOM.compareSurface.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const x = clientX - rect.left;
  return Math.max(0, Math.min(100, (x / rect.width) * 100));
}

function onComparePointerDown(event) {
  event.preventDefault();
  state.isDraggingCompare = true;
  state.comparePosition = getComparePositionFromEvent(event);
  updateCompareSurface();
}

function onComparePointerMove(event) {
  if (!state.isDraggingCompare) return;
  event.preventDefault();
  state.comparePosition = getComparePositionFromEvent(event);
  updateCompareSurface();
}

function onComparePointerUp() {
  state.isDraggingCompare = false;
}

function bindCompareDrag() {
  if (!DOM.compareSurface) return;
  DOM.compareSurface.addEventListener("mousedown", onComparePointerDown);
  DOM.compareSurface.addEventListener("touchstart", onComparePointerDown, { passive: false });
  document.addEventListener("mousemove", onComparePointerMove);
  document.addEventListener("touchmove", onComparePointerMove, { passive: false });
  document.addEventListener("mouseup", onComparePointerUp);
  document.addEventListener("touchend", onComparePointerUp);
}

function renderCanvasMeta() {
  const chunks = [];
  if (state.sourceImage) {
    chunks.push(`<span>source: ${escapeHtml(state.sourceImage.label)}</span>`);
  }
  if (state.activeResult) {
    chunks.push(`<span>${state.activeResult.width}×${state.activeResult.height}</span>`);
    chunks.push(`<span>${state.activeResult.steps} steps</span>`);
    chunks.push(`<span>guidance ${Number(state.activeResult.guidance_scale).toFixed(1)}</span>`);
    chunks.push(`<span>seed ${state.activeResult.seed}</span>`);
  }
  DOM.canvasMeta.innerHTML = chunks.join("");
}

function renderStage() {
  renderCanvasMeta();

  const source = state.sourceImage;
  const active = state.activeResult;
  const canCompare = Boolean(source && active && source.url !== active.url);

  DOM.stagePlaceholder.classList.toggle("hidden", Boolean(source || active));
  DOM.singleStage.classList.toggle("hidden", canCompare || !(source || active));
  DOM.compareStage.classList.toggle("hidden", !canCompare);
  DOM.compareModeBadge.classList.toggle("hidden", !canCompare);
  DOM.fullscreenStageBtn.classList.toggle("hidden", !canCompare);
  DOM.fullscreenStageBtn.querySelector(".btn-label")?.textContent;
  DOM.useActiveAsSourceBtn.classList.toggle("hidden", !active || (source && source.url === active.url));

  if (canCompare) {
    DOM.stageTitle.textContent = "Before / After";
    DOM.compareBeforeImage.src = `${source.url}?t=${Date.now()}`;
    DOM.compareAfterImage.src = `${active.url}?t=${Date.now()}`;
    updateCompareSurface();
    return;
  }

  if (active) {
    DOM.stageTitle.textContent = "Generated";
    DOM.singleStageImage.src = `${active.url}?t=${Date.now()}`;
    DOM.singleStageLabel.textContent = "Generated";
    return;
  }

  if (source) {
    DOM.stageTitle.textContent = "Source";
    DOM.singleStageImage.src = `${source.url}?t=${Date.now()}`;
    DOM.singleStageLabel.textContent = "Source image";
    return;
  }

  DOM.stageTitle.textContent = "Canvas";
}

function clearResults() {
  state.results = [];
  state.activeResult = null;
  DOM.resultStrip.innerHTML = "";
  DOM.resultCount.textContent = "0 images";
  renderStage();
}

function useItemAsSource(item, labelPrefix) {
  setSourceImage({
    url: item.url,
    width: item.width,
    height: item.height,
    label: `${labelPrefix}: ${truncate(item.prompt, 34)}`,
    description: "This image is now the source used for the next generation request.",
    kind: "generation",
  });
}

function renderResultStrip(items) {
  DOM.resultCount.textContent = formatCount(items.length, "image");
  if (!items.length) {
    DOM.resultStrip.innerHTML = "";
    return;
  }

  DOM.resultStrip.innerHTML = items
    .map(
      (item, index) => `
        <article class="thumb-card">
          <button type="button" class="thumb-image-button" data-preview-url="${escapeAttr(item.url)}">
            <img src="${item.url}" alt="Generated image ${index + 1}" />
            <div class="thumb-copy">
              <strong>${escapeHtml(truncate(item.prompt, 42))}</strong>
              <span>${item.width}×${item.height} · seed ${item.seed}</span>
            </div>
          </button>
          <div class="thumb-actions">
            <button type="button" class="btn-inline" data-preview-url="${escapeAttr(item.url)}">Preview</button>
            <button type="button" class="btn-inline" data-source-url="${escapeAttr(item.url)}">Use as source</button>
          </div>
        </article>
      `,
    )
    .join("");

  DOM.resultStrip.querySelectorAll("[data-preview-url]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items.find((entry) => entry.url === button.dataset.previewUrl);
      if (!item) return;
      state.activeResult = item;
      renderStage();
    });
  });

  DOM.resultStrip.querySelectorAll("[data-source-url]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items.find((entry) => entry.url === button.dataset.sourceUrl);
      if (!item) return;
      useItemAsSource(item, "Generated source");
    });
  });
}

function applyHistorySettings(item) {
  DOM.prompt.value = item.prompt || "";
  DOM.negativePrompt.value = item.negative_prompt || "";
  DOM.width.value = item.width;
  DOM.height.value = item.height;
  DOM.presetSelect.value = "native";
  if (state.controlInputs.num_inference_steps) {
    state.controlInputs.num_inference_steps.value = item.steps;
    state.controlInputs.num_inference_steps.dispatchEvent(new Event("input"));
  }
  if (state.controlInputs.guidance_scale) {
    state.controlInputs.guidance_scale.value = item.guidance_scale;
    state.controlInputs.guidance_scale.dispatchEvent(new Event("input"));
  }
  if (state.controlInputs.seed) {
    state.controlInputs.seed.value = item.seed;
  }
  updateResolutionMeta();
}

function renderHistory(items) {
  if (!items.length) {
    DOM.historyList.innerHTML = '<span class="empty-copy">No images saved yet.</span>';
    return;
  }

  DOM.historyList.innerHTML = items
    .map(
      (item) => `
        <article class="history-card">
          <button type="button" class="thumb-image-button" data-history-preview="${escapeAttr(item.id)}">
            <img src="${item.url}" alt="History item" loading="lazy" />
            <div class="history-copy">
              <strong>${escapeHtml(truncate(item.prompt, 42))}</strong>
              <span>${item.width}×${item.height} · seed ${item.seed}</span>
            </div>
          </button>
          <div class="history-actions">
            <button type="button" class="btn-inline" data-history-source="${escapeAttr(item.id)}">Use as source</button>
            <button type="button" class="btn-inline" data-history-apply="${escapeAttr(item.id)}">Load settings</button>
            <button type="button" class="btn-inline danger" data-history-delete="${escapeAttr(item.id)}">Delete</button>
          </div>
        </article>
      `,
    )
    .join("");

  DOM.historyList.querySelectorAll("[data-history-preview]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items.find((entry) => entry.id === button.dataset.historyPreview);
      if (!item) return;
      state.activeResult = item;
      renderStage();
    });
  });

  DOM.historyList.querySelectorAll("[data-history-source]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items.find((entry) => entry.id === button.dataset.historySource);
      if (!item) return;
      useItemAsSource(item, "History source");
    });
  });

  DOM.historyList.querySelectorAll("[data-history-apply]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = items.find((entry) => entry.id === button.dataset.historyApply);
      if (!item) return;
      state.activeResult = item;
      applyHistorySettings(item);
      renderStage();
    });
  });

  DOM.historyList.querySelectorAll("[data-history-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await apiFetch(`/api/history/${button.dataset.historyDelete}`, { method: "DELETE" });
        await loadHistory();
      } catch (error) {
        updateBanner(`Could not delete history item: ${error.message}`, "error");
      }
    });
  });
}

async function loadHistory() {
  try {
    state.history = await apiFetch("/api/history");
    renderHistory(state.history);
  } catch (error) {
    updateBanner(`Could not load history: ${error.message}`, "error");
  }
}

function renderLoraList(loras) {
  if (!loras.length) {
    DOM.loraList.innerHTML = '<span class="empty-copy">No LoRAs loaded.</span>';
    return;
  }

  DOM.loraList.innerHTML = loras
    .map(
      (lora) => `
        <div class="lora-row">
          <div class="lora-row-top">
            <div class="lora-title">
              <strong>${escapeHtml(lora.name)}</strong>
              <span class="lora-path">${escapeHtml(lora.path)}</span>
            </div>
            <div class="card-inline-row">
              <label class="micro-label">
                <input type="checkbox" class="lora-toggle" data-name="${escapeAttr(lora.name)}" ${lora.enabled ? "checked" : ""} />
                enabled
              </label>
              <button type="button" class="btn-inline danger" data-unload-name="${escapeAttr(lora.name)}">Unload</button>
            </div>
          </div>
          <div class="slider-shell">
            <input
              type="range"
              class="lora-strength-slider"
              min="0"
              max="2"
              step="0.05"
              value="${lora.strength}"
              data-name="${escapeAttr(lora.name)}"
            />
            <span class="slider-value">${Number(lora.strength).toFixed(2)}</span>
          </div>
        </div>
      `,
    )
    .join("");

  DOM.loraList.querySelectorAll(".lora-toggle").forEach((toggle) => {
    toggle.addEventListener("change", async () => {
      try {
        const result = await apiFetch("/api/loras/toggle", {
          method: "POST",
          body: JSON.stringify({ name: toggle.dataset.name, enabled: toggle.checked }),
        });
        renderLoraList(result.loras);
      } catch (error) {
        updateBanner(`Could not toggle LoRA: ${error.message}`, "error");
      }
    });
  });

  DOM.loraList.querySelectorAll(".lora-strength-slider").forEach((slider) => {
    slider.addEventListener("input", () => {
      slider.nextElementSibling.textContent = Number(slider.value).toFixed(2);
    });
    slider.addEventListener("change", async () => {
      try {
        const result = await apiFetch("/api/loras/strength", {
          method: "POST",
          body: JSON.stringify({ name: slider.dataset.name, strength: Number(slider.value) }),
        });
        renderLoraList(result.loras);
      } catch (error) {
        updateBanner(`Could not update LoRA strength: ${error.message}`, "error");
      }
    });
  });

  DOM.loraList.querySelectorAll("[data-unload-name]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const result = await apiFetch("/api/loras/unload", {
          method: "POST",
          body: JSON.stringify({ name: button.dataset.unloadName }),
        });
        renderLoraList(result.loras);
      } catch (error) {
        updateBanner(`Could not unload LoRA: ${error.message}`, "error");
      }
    });
  });
}

async function loadAvailableLoras() {
  try {
    const loras = await apiFetch("/api/loras/available");
    if (!loras.length) {
      DOM.availableLorasList.innerHTML = '<span class="empty-copy">No local adapters found.</span>';
      return;
    }

    DOM.availableLorasList.innerHTML = loras
      .map(
        (lora) => `
          <div class="available-lora-row">
            <div class="lora-title">
              <strong>${escapeHtml(lora.name)}</strong>
              <span class="lora-path">${escapeHtml(lora.path)}</span>
            </div>
            <button type="button" class="btn-inline" data-fill-name="${escapeAttr(lora.name)}" data-fill-path="${escapeAttr(lora.path)}">
              Use
            </button>
          </div>
        `,
      )
      .join("");

    DOM.availableLorasList.querySelectorAll("[data-fill-name]").forEach((button) => {
      button.addEventListener("click", () => {
        DOM.loraNameInput.value = button.dataset.fillName;
        DOM.loraPathInput.value = button.dataset.fillPath;
      });
    });
  } catch (error) {
    updateBanner(`Could not scan local adapters: ${error.message}`, "error");
  }
}

async function handleAddLora() {
  const name = DOM.loraNameInput.value.trim();
  const path = DOM.loraPathInput.value.trim();
  const strength = Number(DOM.loraStrengthInput.value);
  if (!name || !path) {
    updateBanner("LoRA name and path are both required.", "warning");
    return;
  }

  try {
    const result = await apiFetch("/api/loras/load", {
      method: "POST",
      body: JSON.stringify({ name, path, strength }),
    });
    renderLoraList(result.loras);
    DOM.loraNameInput.value = "";
    DOM.loraPathInput.value = "";
  } catch (error) {
    updateBanner(`Could not load LoRA: ${error.message}`, "error");
  }
}

async function handleGenerate() {
  const payload = generationPayload();
  if (!payload.prompt) {
    DOM.prompt.focus();
    updateBanner("Prompt is required before generation.", "warning");
    return;
  }

  try {
    setGenerating(true);
    const result = await apiFetch("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.results = result.images || [];
    state.activeResult = state.results[0] || null;
    renderResultStrip(state.results);
    renderStage();
    await loadHistory();
  } catch (error) {
    updateBanner(`Generation failed: ${error.message}`, "error");
  } finally {
    setGenerating(false);
  }
}

async function handleImageUpload() {
  const file = DOM.imageUpload.files?.[0];
  if (!file) return;

  try {
    const formData = new FormData();
    formData.append("file", file);
    const result = await apiFetch("/api/upload-image", { method: "POST", body: formData });
    DOM.width.value = result.width;
    DOM.height.value = result.height;
    DOM.presetSelect.value = "native";
    updateResolutionMeta();
    setSourceImage({
      url: result.url,
      width: result.width,
      height: result.height,
      label: file.name,
      description: "Uploaded source image. This remains the generation input until you promote another image.",
      kind: "upload",
    });
  } catch (error) {
    updateBanner(`Upload failed: ${error.message}`, "error");
  } finally {
    DOM.imageUpload.value = "";
  }
}

async function handleInitPipeline() {
  try {
    const status = await apiFetch("/api/pipeline/initialize", { method: "POST" });
    renderStatus(status);
    if (status.ready) {
      renderLoraList(status.loras || []);
      await loadAvailableLoras();
    }
  } catch (error) {
    updateBanner(`Could not initialize the pipeline: ${error.message}`, "error");
  }
}

async function refreshStatus() {
  try {
    const status = await apiFetch("/api/status");
    renderStatus(status);
    if (status.ready) {
      renderLoraList(status.loras || []);
    }
  } catch (error) {
    updateBanner(`Could not reach the backend: ${error.message}`, "error");
  }
}

async function loadConfig() {
  state.config = await apiFetch("/api/config");
  renderControls();
  renderPresets();
  applyConfigDefaults();
  renderStatus(state.config.status);
}

/* ─── Collapsible panels ─── */
function initPanels() {
  document.querySelectorAll(".panel[data-panel]").forEach((panel) => {
    // Open prompt and source by default, collapse the rest
    const name = panel.dataset.panel;
    if (name === "prompt" || name === "source" || name === "canvas") {
      panel.classList.add("open");
    }

    const toggle = panel.querySelector(".panel-toggle");
    if (!toggle) return;
    toggle.addEventListener("click", () => {
      panel.classList.toggle("open");
    });
  });
}

function bindEvents() {
  DOM.width.addEventListener("input", updateResolutionMeta);
  DOM.height.addEventListener("input", updateResolutionMeta);
  DOM.presetSelect.addEventListener("change", updateResolutionMeta);
  DOM.imageUpload.addEventListener("change", handleImageUpload);
  DOM.generateBtn.addEventListener("click", handleGenerate);
  DOM.clearResultsBtn.addEventListener("click", clearResults);
  DOM.clearSourceBtn.addEventListener("click", () => setSourceImage(null));
  DOM.initPipelineBtn.addEventListener("click", handleInitPipeline);
  DOM.loraAddBtn.addEventListener("click", handleAddLora);

  // Hidden range kept in sync via drag — but also listen for direct changes
  DOM.compareSlider.addEventListener("input", () => {
    state.comparePosition = Number(DOM.compareSlider.value);
    updateCompareSurface();
  });

  DOM.fullscreenStageBtn.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement === DOM.stagePanel) {
        await document.exitFullscreen();
      } else {
        await DOM.stagePanel.requestFullscreen();
      }
      renderStage();
    } catch (error) {
      updateBanner(`Could not toggle fullscreen: ${error.message}`, "error");
    }
  });

  DOM.useActiveAsSourceBtn.addEventListener("click", () => {
    if (!state.activeResult) return;
    useItemAsSource(state.activeResult, "Generated source");
  });

  DOM.loraStrengthInput.addEventListener("input", () => {
    DOM.loraStrengthValue.textContent = Number(DOM.loraStrengthInput.value).toFixed(2);
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      handleGenerate();
    }
  });
  document.addEventListener("fullscreenchange", renderStage);

  // Bind interactive on-image compare drag
  bindCompareDrag();
}

async function init() {
  initPanels();
  bindEvents();
  clearResults();
  renderSourceCard();

  try {
    await loadConfig();
    await loadHistory();
    await loadAvailableLoras();
    if (state.status?.ready) {
      renderLoraList(state.status.loras || []);
    }
  } catch (error) {
    updateBanner(`Startup failed: ${error.message}`, "error");
  }

  setInterval(refreshStatus, 10000);
}

document.addEventListener("DOMContentLoaded", init);
