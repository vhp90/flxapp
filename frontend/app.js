/* ================================================================
   Flux2 GUI – Application Logic
   ================================================================ */

const API = '';

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
  status: $('#pipeline-status'),
  prompt: $('#prompt-input'),
  negPrompt: $('#negative-prompt-input'),
  presetSelect: $('#preset-select'),
  width: $('#width-input'),
  height: $('#height-input'),
  megapixelDisplay: $('#megapixel-display'),
  aspectDisplay: $('#aspect-display'),
  outputWidth: $('#output-width'),
  outputHeight: $('#output-height'),
  outputMp: $('#output-mp'),
  imageUpload: $('#image-upload'),
  stepsSlider: $('#steps-slider'),
  stepsValue: $('#steps-value'),
  guidanceSlider: $('#guidance-slider'),
  guidanceValue: $('#guidance-value'),
  seed: $('#seed-input'),
  numImages: $('#num-images-input'),
  generateBtn: $('#generate-btn'),
  generateSpinner: $('#generate-spinner'),
  previewImg: $('#preview-image'),
  previewPlaceholder: $('#preview-placeholder'),
  previewMeta: $('#preview-meta'),
  metaResolution: $('#meta-resolution'),
  metaSteps: $('#meta-steps'),
  metaSeed: $('#meta-seed'),
  loraList: $('#lora-list'),
  loraNameInput: $('#lora-name-input'),
  loraPathInput: $('#lora-path-input'),
  loraStrengthInput: $('#lora-strength-input'),
  loraStrengthValue: $('#lora-strength-value'),
  loraAddBtn: $('#lora-add-btn'),
  availableLorasList: $('#available-loras-list'),
  historyList: $('#history-list'),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isGenerating = false;
let presets = [];

// ---------------------------------------------------------------------------
// Utility – API calls
// ---------------------------------------------------------------------------
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'API error');
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Pipeline Status
// ---------------------------------------------------------------------------
async function checkStatus() {
  try {
    const data = await apiFetch('/api/status');
    DOM.status.textContent = data.ready ? 'Online' : 'Loading…';
    DOM.status.className = `status-badge ${data.ready ? 'online' : 'offline'}`;
    if (data.ready) {
      renderLoraList(data.loras);
    }
  } catch {
    DOM.status.textContent = 'Offline';
    DOM.status.className = 'status-badge offline';
  }
}

// ---------------------------------------------------------------------------
// Slider sync (real-time value display)
// ---------------------------------------------------------------------------
function syncSliders() {
  DOM.stepsSlider.addEventListener('input', () => {
    DOM.stepsValue.textContent = DOM.stepsSlider.value;
  });
  DOM.guidanceSlider.addEventListener('input', () => {
    DOM.guidanceValue.textContent = parseFloat(DOM.guidanceSlider.value).toFixed(1);
  });
  DOM.loraStrengthInput.addEventListener('input', () => {
    DOM.loraStrengthValue.textContent = parseFloat(DOM.loraStrengthInput.value).toFixed(2);
  });
}

// ---------------------------------------------------------------------------
// Resolution helpers
// ---------------------------------------------------------------------------
function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }

/** Round to nearest multiple of 64, clamped between 256 and 4096 */
function roundTo64(v) {
  return Math.max(256, Math.min(4096, Math.round(v / 64) * 64));
}

/** Calculate output width/height by scaling base aspect ratio to target megapixels */
function calcOutputDims(baseW, baseH, targetMP) {
  const aspect = baseW / baseH;
  // targetMP * 1e6 = outW * outH, and outW/outH = aspect
  // outH = sqrt(targetMP*1e6 / aspect),  outW = outH * aspect
  const outH = Math.sqrt((targetMP * 1_000_000) / aspect);
  const outW = outH * aspect;
  return { w: roundTo64(outW), h: roundTo64(outH) };
}

function updateResolutionMeta() {
  const w = parseInt(DOM.width.value, 10) || 1024;
  const h = parseInt(DOM.height.value, 10) || 1024;
  const mp = (w * h) / 1_000_000;
  DOM.megapixelDisplay.textContent = `${mp.toFixed(2)} MP`;
  const g = gcd(w, h);
  DOM.aspectDisplay.textContent = `${w / g}:${h / g}`;

  // Recalculate output dims
  updateOutputDims();
}

function updateOutputDims() {
  const baseW = parseInt(DOM.width.value, 10) || 1024;
  const baseH = parseInt(DOM.height.value, 10) || 1024;
  const selected = DOM.presetSelect.value;

  let outW, outH;
  if (selected === 'native') {
    outW = baseW;
    outH = baseH;
  } else {
    const targetMP = parseFloat(selected);
    const dims = calcOutputDims(baseW, baseH, targetMP);
    outW = dims.w;
    outH = dims.h;
  }

  DOM.outputWidth.textContent = outW;
  DOM.outputHeight.textContent = outH;
  const outMP = (outW * outH) / 1_000_000;
  DOM.outputMp.textContent = `${outMP.toFixed(2)} MP`;
}

function syncResolutionInputs() {
  DOM.width.addEventListener('input', updateResolutionMeta);
  DOM.height.addEventListener('input', updateResolutionMeta);
}

// ---------------------------------------------------------------------------
// Quality Presets (MP targets)
// ---------------------------------------------------------------------------
async function loadPresets() {
  try {
    presets = await apiFetch('/api/presets');
    presets.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.megapixels;
      opt.textContent = `Upscale to ${p.name}`;
      DOM.presetSelect.appendChild(opt);
    });
  } catch {
    // presets endpoint not available
  }
}

function syncPresetSelect() {
  DOM.presetSelect.addEventListener('change', updateOutputDims);
}

// ---------------------------------------------------------------------------
// Image Upload – Auto Dimensions
// ---------------------------------------------------------------------------
function syncImageUpload() {
  DOM.imageUpload.addEventListener('change', async () => {
    const file = DOM.imageUpload.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload-image', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();

      // Set base dimensions from uploaded image
      DOM.width.value = data.width;
      DOM.height.value = data.height;
      updateResolutionMeta();

      // Show uploaded image in preview
      showPreview(data.url, {
        width: data.width,
        height: data.height,
        num_inference_steps: '-',
        seed: '-',
      });
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    }

    DOM.imageUpload.value = '';
  });
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------
function getOutputDims() {
  const baseW = parseInt(DOM.width.value, 10) || 1024;
  const baseH = parseInt(DOM.height.value, 10) || 1024;
  const selected = DOM.presetSelect.value;

  if (selected === 'native') {
    return { w: baseW, h: baseH };
  }
  const targetMP = parseFloat(selected);
  return calcOutputDims(baseW, baseH, targetMP);
}

async function handleGenerate() {
  if (isGenerating) return;
  const prompt = DOM.prompt.value.trim();
  if (!prompt) { DOM.prompt.focus(); return; }

  isGenerating = true;
  DOM.generateBtn.disabled = true;
  DOM.generateBtn.querySelector('.btn-label').textContent = 'Generating…';
  DOM.generateSpinner.classList.remove('hidden');

  const dims = getOutputDims();

  try {
    const payload = {
      prompt,
      negative_prompt: DOM.negPrompt.value.trim(),
      width: dims.w,
      height: dims.h,
      num_inference_steps: parseInt(DOM.stepsSlider.value, 10),
      guidance_scale: parseFloat(DOM.guidanceSlider.value),
      seed: parseInt(DOM.seed.value, 10),
      num_images: parseInt(DOM.numImages.value, 10),
    };

    const data = await apiFetch('/api/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    if (data.images && data.images.length > 0) {
      showPreview(data.images[0], payload);
    }

    await loadHistory();
  } catch (err) {
    alert(`Generation failed: ${err.message}`);
  } finally {
    isGenerating = false;
    DOM.generateBtn.disabled = false;
    DOM.generateBtn.querySelector('.btn-label').textContent = 'Generate';
    DOM.generateSpinner.classList.add('hidden');
  }
}

function showPreview(url, params) {
  DOM.previewPlaceholder.classList.add('hidden');
  DOM.previewImg.classList.remove('hidden');
  DOM.previewImg.src = `${url}?t=${Date.now()}`;
  DOM.previewMeta.classList.remove('hidden');
  DOM.metaResolution.textContent = `${params.width}×${params.height}`;
  DOM.metaSteps.textContent = `${params.num_inference_steps} steps`;
  DOM.metaSeed.textContent = `seed: ${params.seed}`;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
async function loadHistory() {
  try {
    const history = await apiFetch('/api/history');
    renderHistory(history);
  } catch {
    DOM.historyList.innerHTML = '<span class="muted">Failed to load history</span>';
  }
}

function renderHistory(items) {
  if (!items || items.length === 0) {
    DOM.historyList.innerHTML = '<span class="muted">No images yet</span>';
    return;
  }

  DOM.historyList.innerHTML = items.map((item) => `
    <div class="history-card" data-id="${item.id}" title="${escapeHtml(item.prompt)}">
      <img src="${item.url}" alt="Generated" loading="lazy" />
      <div class="history-overlay">${escapeHtml(truncate(item.prompt, 40))}</div>
      <button class="history-delete" data-id="${item.id}" title="Delete">✕</button>
    </div>
  `).join('');

  DOM.historyList.querySelectorAll('.history-card').forEach((card) => {
    card.addEventListener('click', (e) => {
      if (e.target.classList.contains('history-delete')) return;
      const item = items.find((i) => i.id === card.dataset.id);
      if (item) {
        showPreview(item.url, {
          width: item.width,
          height: item.height,
          num_inference_steps: item.steps,
          seed: item.seed,
        });
        DOM.width.value = item.width;
        DOM.height.value = item.height;
        DOM.presetSelect.value = 'native';
        updateResolutionMeta();
      }
    });
  });

  DOM.historyList.querySelectorAll('.history-delete').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        await apiFetch(`/api/history/${btn.dataset.id}`, { method: 'DELETE' });
        await loadHistory();
      } catch (err) {
        console.error('Delete failed:', err);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// LoRA Management
// ---------------------------------------------------------------------------
function renderLoraList(loras) {
  if (!loras || loras.length === 0) {
    DOM.loraList.innerHTML = '<span class="muted">No LoRAs loaded</span>';
    return;
  }

  DOM.loraList.innerHTML = loras.map((lora) => `
    <div class="lora-card ${lora.enabled ? '' : 'disabled'}">
      <input type="checkbox" class="lora-toggle" data-name="${escapeAttr(lora.name)}" ${lora.enabled ? 'checked' : ''} />
      <span class="lora-name" title="${escapeAttr(lora.path)}">${escapeHtml(lora.name)}</span>
      <input type="range" class="lora-strength-slider" data-name="${escapeAttr(lora.name)}" min="0" max="2" step="0.05" value="${lora.strength}" />
      <span class="lora-strength-val">${lora.strength.toFixed(2)}</span>
      <button class="btn-danger" data-name="${escapeAttr(lora.name)}" title="Unload">✕</button>
    </div>
  `).join('');

  DOM.loraList.querySelectorAll('.lora-toggle').forEach((toggle) => {
    toggle.addEventListener('change', async () => {
      try {
        const res = await apiFetch('/api/loras/toggle', {
          method: 'POST',
          body: JSON.stringify({ name: toggle.dataset.name, enabled: toggle.checked }),
        });
        renderLoraList(res.loras);
      } catch (err) { alert(`Toggle failed: ${err.message}`); }
    });
  });

  DOM.loraList.querySelectorAll('.lora-strength-slider').forEach((slider) => {
    slider.addEventListener('change', async () => {
      try {
        const res = await apiFetch('/api/loras/strength', {
          method: 'POST',
          body: JSON.stringify({ name: slider.dataset.name, strength: parseFloat(slider.value) }),
        });
        renderLoraList(res.loras);
      } catch (err) { alert(`Strength update failed: ${err.message}`); }
    });
    slider.addEventListener('input', () => {
      slider.nextElementSibling.textContent = parseFloat(slider.value).toFixed(2);
    });
  });

  DOM.loraList.querySelectorAll('.btn-danger').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        const res = await apiFetch('/api/loras/unload', {
          method: 'POST',
          body: JSON.stringify({ name: btn.dataset.name }),
        });
        renderLoraList(res.loras);
      } catch (err) { alert(`Unload failed: ${err.message}`); }
    });
  });
}

async function handleAddLora() {
  const name = DOM.loraNameInput.value.trim();
  const path = DOM.loraPathInput.value.trim();
  const strength = parseFloat(DOM.loraStrengthInput.value);
  if (!name || !path) { alert('Provide both a name and path for the LoRA.'); return; }

  try {
    const res = await apiFetch('/api/loras/load', {
      method: 'POST',
      body: JSON.stringify({ name, path, strength }),
    });
    renderLoraList(res.loras);
    DOM.loraNameInput.value = '';
    DOM.loraPathInput.value = '';
    DOM.loraStrengthInput.value = '1.0';
    DOM.loraStrengthValue.textContent = '1.00';
  } catch (err) {
    alert(`Failed to load LoRA: ${err.message}`);
  }
}

async function loadAvailableLoras() {
  try {
    const available = await apiFetch('/api/loras/available');
    if (available.length === 0) {
      DOM.availableLorasList.innerHTML = '<span class="muted">No .safetensors files in loras/ folder</span>';
      return;
    }
    DOM.availableLorasList.innerHTML = available.map((l) => `
      <div class="lora-available-item">
        <span>${escapeHtml(l.name)}</span>
        <button data-name="${escapeAttr(l.name)}" data-path="${escapeAttr(l.path)}">Load</button>
      </div>
    `).join('');

    DOM.availableLorasList.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        DOM.loraNameInput.value = btn.dataset.name;
        DOM.loraPathInput.value = btn.dataset.path;
      });
    });
  } catch {
    DOM.availableLorasList.innerHTML = '<span class="muted">Could not scan loras directory</span>';
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len) + '…' : str;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  syncSliders();
  syncResolutionInputs();
  syncPresetSelect();
  syncImageUpload();
  updateResolutionMeta();

  DOM.generateBtn.addEventListener('click', handleGenerate);
  DOM.loraAddBtn.addEventListener('click', handleAddLora);

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleGenerate();
  });

  checkStatus();
  loadPresets();
  loadHistory();
  loadAvailableLoras();

  setInterval(checkStatus, 10000);
}

document.addEventListener('DOMContentLoaded', init);
