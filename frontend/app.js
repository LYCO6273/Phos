const state = {
  // grainEnabled: true, // Removed
  files: [],
  presets: null,
  currentFilm: "FUJI200",
  processedBlobUrl: null,
  originalBlobUrl: null,
  progress: {
    rafId: null,
    startTs: 0,
    value: 0,
  },
  job: {
    id: null,
    controller: null,
  },
};

// --- Utils ---
function $(id) {
  return document.getElementById(id);
}

function setStatus(text, isError = false) {
  const el = document.querySelector(".status-indicator");
  if (el) {
    el.innerHTML = `<span class="status-dot" style="background-color: ${isError ? 'var(--accent-safe-red)' : 'var(--accent-active)'}"></span> ${text}`;
  }
}

function formatBytesDecimal(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1000) return `${n} B`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)} KB`;
  if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(2)} MB`;
  return `${(n / 1_000_000_000).toFixed(2)} GB`;
}

function fmt(n) {
  const v = Number(n);
  if (Number.isNaN(v)) return "";
  return v.toFixed(2).replace(/\.00$/, "");
}

function hexToRgba(hex, alpha = 0.3) {
  const c = String(hex || "").replace("#", "").trim();
  if (c.length !== 6) return `rgba(255, 193, 7, ${alpha})`;
  const r = parseInt(c.slice(0, 2), 16);
  const g = parseInt(c.slice(2, 4), 16);
  const b = parseInt(c.slice(4, 6), 16);
  if ([r, g, b].some(Number.isNaN)) return `rgba(255, 193, 7, ${alpha})`;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function stripExtension(filename) {
  const name = String(filename || "image");
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

function filenameFromContentDisposition(contentDisposition) {
  const cd = String(contentDisposition || "");
  // RFC 5987: filename*=UTF-8''...
  const star = cd.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  if (star && star[1]) {
    try {
      return decodeURIComponent(star[1].trim());
    } catch {
      return star[1].trim();
    }
  }

  const plain = cd.match(/filename\s*=\s*\"?([^\";]+)\"?/i);
  if (plain && plain[1]) return plain[1].trim();
  return "";
}

function resetSelection() {
  state.files = [];
  cancelCurrentJob();
  if (state.processedBlobUrl) URL.revokeObjectURL(state.processedBlobUrl);
  if (state.originalBlobUrl) URL.revokeObjectURL(state.originalBlobUrl);
  state.processedBlobUrl = null;
  state.originalBlobUrl = null;

  $("imageMeta").textContent = "未加载图片";
  const promptText = document.querySelector(".upload-prompt p");
  if (promptText) promptText.textContent = "打开图片";

  const prompt = $("uploadPrompt");
  const compare = $("compareContainer");
  compare.classList.add("hidden");
  compare.classList.remove("visible");
  prompt.style.display = "";
  prompt.style.opacity = "1";

  const fileInput = $("files");
  if (fileInput) fileInput.value = "";

  stopProgress();
  $("btnDownload").disabled = true;
  $("btnReset").disabled = true;
  setStatus("请选择图片");
}

function initProgressUI() {
  const blocks = $("progressBlocks");
  if (!blocks) return;
  blocks.innerHTML = "";
  const total = 24;
  for (let i = 0; i < total; i += 1) {
    const block = document.createElement("div");
    block.className = "progress-block";
    block.dataset.index = String(i);
    blocks.appendChild(block);
  }
}

function setProgress(pct) {
  const wrap = $("progressWrap");
  const blocks = $("progressBlocks");
  const text = $("progressText");
  if (!wrap || !blocks || !text) return;

  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  state.progress.value = value;
  text.textContent = `${Math.floor(value)}%`;

  const nodes = blocks.querySelectorAll(".progress-block");
  const activeCount = Math.round((value / 100) * nodes.length);
  nodes.forEach((node, idx) => {
    node.classList.toggle("active", idx < activeCount);
    node.classList.toggle("pulse", idx === Math.min(activeCount, nodes.length - 1) && value < 100);
  });
}

function startProgress() {
  const wrap = $("progressWrap");
  if (!wrap) return;
  wrap.classList.remove("hidden");
  // Ensure transition applies.
  requestAnimationFrame(() => wrap.classList.add("visible"));

  setProgress(0);
}

function stopProgress() {
  const wrap = $("progressWrap");
  if (!wrap) return;
  state.progress.rafId = null;
  state.progress.startTs = 0;
  state.progress.value = 0;
  wrap.classList.remove("visible");
  setTimeout(() => {
    wrap.classList.add("hidden");
    setProgress(0);
  }, 250);
}

function finishProgress() {
  setProgress(100);
  setTimeout(() => stopProgress(), 450);
}

function cancelCurrentJob() {
  if (state.job.controller) {
    try { state.job.controller.abort(); } catch {}
  }
  state.job.controller = null;
  state.job.id = null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function createJob(file, options, signal) {
  const fd = new FormData();
  fd.append("file", file);
  Object.entries(options).forEach(([k, v]) => fd.append(k, String(v)));

  const res = await fetch("/api/jobs", { method: "POST", body: fd, signal });
  if (!res.ok) {
    let detail = `创建任务失败 (${res.status})`;
    try {
      const data = await res.json();
      if (data && typeof data.detail !== "undefined") detail = String(data.detail);
    } catch {
      try {
        const text = await res.text();
        if (text) detail = text;
      } catch {}
    }
    throw new Error(detail);
  }

  const data = await res.json();
  const jobId = data?.job_id;
  if (!jobId) throw new Error("任务创建失败：缺少 job_id");
  return jobId;
}

async function pollJob(jobId, signal) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
  if (!res.ok) throw new Error(`获取进度失败 (${res.status})`);
  return res.json();
}

async function fetchJobResult(jobId, signal) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`, { signal });
  if (!res.ok) {
    let detail = `获取结果失败 (${res.status})`;
    try {
      const data = await res.json();
      if (data && typeof data.detail !== "undefined") detail = String(data.detail);
    } catch {
      try {
        const text = await res.text();
        if (text) detail = text;
      } catch {}
    }
    throw new Error(detail);
  }
  return res;
}

// --- API ---
async function loadPresets() {
  const res = await fetch("/api/presets");
  if (!res.ok) throw new Error("无法加载胶卷预设");
  const data = await res.json();
  return {
    film_types: data.film_types || [],
    meta: data.film_meta || {},
    default_film_type: data.default_film_type || "FUJI200",
  };
}

function readOptions() {
  const grainStrength = Number($("grainStrength")?.value ?? 1);
  const grain_enabled = grainStrength > 0.001;
  return {
    film_type: state.currentFilm,
    tone_style: "filmic",
    grain_enabled,
    grain_strength: grainStrength,
    grain_size: 1.0,
    jpeg_quality: 95,
  };
}

// Film Selector with Carousel
let currentIndex = 0;
let carouselOffset = 0;
let isDragging = false;
let startX = 0;
let currentX = 0;

function renderFilmShelf(presets) {
  const track = $("carouselTrack");
  track.innerHTML = "";

  presets.film_types.forEach((type, index) => {
    const info = presets.meta[type];
    if (!info) {
      console.warn("Missing film meta:", type);
      return;
    }
    const el = document.createElement("div");
    el.className = `film-canister ${type === state.currentFilm ? "active selected" : ""}`;
    el.dataset.film = type;
    el.dataset.index = index;
    el.style.setProperty("--film-glow", hexToRgba(info.brandColor, 0.35));

    // Tooltip Structure
    const tooltip = `
      <div class="film-tooltip">
        <div class="tt-header" style="color: ${info.brandColor}">${info.name}</div>
        <div class="tt-body">
          <div class="tt-row"><span>感光度</span> ${info.iso}</div>
          <div class="tt-row"><span>天气</span> ${info.weather}</div>
          <div class="tt-row"><span>场景</span> ${info.scene}</div>
          <div class="tt-desc">${info.desc}</div>
        </div>
      </div>
    `;

    // Info Overlay
    const infoOverlay = `
      <div class="film-info-overlay">
        <div class="film-name-label">${info.name.split(' ')[0]}</div>
        <div class="film-type-label">${info.type}</div>
      </div>
    `;

    el.innerHTML = `
      <img src="assets/film-cans/${info.image}" alt="${info.name}" class="film-canister-img">
      ${infoOverlay}
      ${tooltip}
    `;

    el.addEventListener("click", () => selectFilm(type, index));
    track.appendChild(el);
  });

  updateFilmInfo();
  updateCarouselPosition();
  initCarouselDrag();
}

function selectFilm(type, index) {
  state.currentFilm = type;
  currentIndex = index;

  // Update active states
  document.querySelectorAll(".film-canister").forEach(el => {
    el.classList.toggle("active", el.dataset.film === type);
    el.classList.toggle("selected", el.dataset.film === type);
  });

  $("currentFilm").textContent = type;
  updateFilmInfo();
  centerOnActive();
}

function updateFilmInfo() {
  if (!state.presets) return;
  const info = state.presets.meta[state.currentFilm];
  if (!info) return;

  $("currentFilm").textContent = state.currentFilm;
  $("infoType").textContent = info.type;
  $("infoIso").textContent = info.iso;
  $("infoRec").textContent = info.scene;
}

// Carousel Navigation
function initCarouselDrag() {
  const track = $("carouselTrack");

  // Mouse Events
  track.addEventListener("mousedown", startDrag);
  window.addEventListener("mousemove", drag);
  window.addEventListener("mouseup", endDrag);

  // Touch Events
  track.addEventListener("touchstart", startDrag);
  window.addEventListener("touchmove", drag);
  window.addEventListener("touchend", endDrag);

  // Navigation Buttons
  $("prevBtn").addEventListener("click", () => navigateCarousel(-1));
  $("nextBtn").addEventListener("click", () => navigateCarousel(1));
}

function startDrag(e) {
  isDragging = true;
  startX = e.type === "mousedown" ? e.clientX : e.touches[0].clientX;
  currentX = startX;
  $("carouselTrack").classList.add("dragging");
}

function drag(e) {
  if (!isDragging) return;
  e.preventDefault();
  currentX = e.type === "mousemove" ? e.clientX : e.touches[0].clientX;
  const diffX = currentX - startX;
  updateCarouselPosition(diffX);
}

function endDrag() {
  if (!isDragging) return;
  isDragging = false;
  $("carouselTrack").classList.remove("dragging");

  const diffX = currentX - startX;
  const threshold = 50;

  if (Math.abs(diffX) > threshold) {
    if (diffX > 0) {
      navigateCarousel(-1);
    } else {
      navigateCarousel(1);
    }
  } else {
    // Snap back
    updateCarouselPosition();
  }
}

function navigateCarousel(direction) {
  const totalFilms = state.presets.film_types.length;
  currentIndex = (currentIndex + direction + totalFilms) % totalFilms;
  selectFilm(state.presets.film_types[currentIndex], currentIndex);
}

function updateCarouselPosition(overrideOffset = 0) {
  const track = $("carouselTrack");
  const canisterWidth = 100;
  const gap = 30;
  const padding = 60;
  const containerWidth = 340;
  const visibleWidth = containerWidth - padding * 2;

  // Calculate offset to center active item
  const activeCanister = document.querySelector(".film-canister.active");
  if (!activeCanister) return;

  const activeIndex = parseInt(activeCanister.dataset.index);
  // Use canisterWidth directly since transform scale doesn't affect layout flow
  const activePosition = padding + (canisterWidth / 2) + activeIndex * (canisterWidth + gap);

  carouselOffset = overrideOffset || (containerWidth / 2 - activePosition);

  // Clamp offset to prevent empty space
  // Relaxed clamping to allow centering the first and last items
  // const maxOffset = containerWidth / 2 - (padding + canisterWidth / 2);
  // const minOffset = containerWidth / 2 - (padding + (state.presets.film_types.length - 1) * (canisterWidth + gap) + canisterWidth / 2);

  // Actually, for a "center focused" carousel, we often don't want strict clamping if we want the active item ALWAYS centered.
  // But if we want to behave like a scrollable list:
  // Let's just remove clamping for now to ensure centering works, or adjust it.
  // The original code had clamping. Let's stick to "always center active" which is simpler and often looks better for this type of selector.

  // carouselOffset = Math.max(minOffset, Math.min(maxOffset, carouselOffset));

  track.style.transform = `translateX(${carouselOffset}px)`;
}

function centerOnActive() {
  updateCarouselPosition();
}

// Comparison Slider
function initComparisonSlider() {
  const container = $("compareContainer");
  const handle = $("compareHandle");
  const processedImg = $("imgProcessed");
  let isDragging = false;

  const updateSlider = (clientX) => {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    handle.style.left = `${percent}%`;
    processedImg.style.clipPath = `inset(0 0 0 ${percent}%)`;
  };

  handle.addEventListener("mousedown", (e) => {
    isDragging = true;
    e.preventDefault(); // Prevent text selection
  });

  window.addEventListener("mouseup", () => isDragging = false);

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    updateSlider(e.clientX);
  });

  // Touch support
  handle.addEventListener("touchstart", (e) => {
    isDragging = true;
  });
  window.addEventListener("touchend", () => isDragging = false);
  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    updateSlider(e.touches[0].clientX);
  });
}

// Processing
async function processImage() {
  if (!state.files.length) return;

  const file = state.files[0];
  setStatus("正在冲洗...");

  $("btnProcess").disabled = true;
  $("btnReset").disabled = false;
  startProgress();

  // Add loading state to button
  const originalBtnText = $("btnProcess").textContent;
  $("btnProcess").textContent = "处理中...";

  try {
    const options = readOptions();

    cancelCurrentJob();
    const controller = new AbortController();
    state.job.controller = controller;
    const signal = controller.signal;

    const jobId = await createJob(file, options, signal);
    state.job.id = jobId;

    while (true) {
      const job = await pollJob(jobId, signal);
      const progress = Number(job?.progress ?? 0);
      const message = String(job?.message ?? "");
      const status = String(job?.status ?? "");
      if (Number.isFinite(progress)) setProgress(progress);
      if (message) setStatus(message);

      if (status === "done") break;
      if (status === "error") throw new Error(job?.error || job?.message || "冲洗失败");
      await sleep(250);
    }

    finishProgress();
    const res = await fetchJobResult(jobId, signal);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    state.processedBlobUrl = url;
    state.originalBlobUrl = URL.createObjectURL(file);

    $("imgOriginal").style.backgroundImage = `url(${state.originalBlobUrl})`;
    $("imgProcessed").style.backgroundImage = `url(${state.processedBlobUrl})`;

    // Transition
    const prompt = $("uploadPrompt");
    const compare = $("compareContainer");

    prompt.style.opacity = '0';
    setTimeout(() => {
      prompt.style.display = "none";
      compare.classList.remove("hidden");
      compare.classList.add("visible");
    }, 300);

    $("imageMeta").textContent = `${file.name} | ${formatBytesDecimal(file.size)}`;

    $("btnDownload").disabled = false;
    $("btnReset").disabled = false;
    $("btnDownload").onclick = () => {
      const a = document.createElement("a");
      a.href = url;
      const suggested = filenameFromContentDisposition(res.headers.get("content-disposition"));
      a.download = suggested || `phos_${state.currentFilm}_${stripExtension(file.name)}.jpg`;
      a.click();
    };

    setStatus("冲洗完成");

  } catch (e) {
    if (e?.name === "AbortError") {
      setStatus("已取消");
      stopProgress();
      return;
    }
    setStatus(`错误: ${e.message}`, true);
    stopProgress();
  } finally {
    cancelCurrentJob();
    $("btnProcess").disabled = false;
    $("btnProcess").textContent = originalBtnText;
  }
}

// Drag and Drop
function initDragDrop() {
  const dropZone = $("uploadPrompt");
  const fileInput = $("files");

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
  });

  dropZone.addEventListener('drop', handleDrop, false);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
  }

  fileInput.addEventListener('change', function () {
    handleFiles(this.files);
  });
}

function handleFiles(files) {
  if (files.length) {
    // If we already have a processed view, switch back to upload state first.
    if (state.processedBlobUrl || state.originalBlobUrl) {
      resetSelection();
    }
    state.files = Array.from(files);
    setStatus("胶卷已加载");
    $("imageMeta").textContent = `${state.files[0].name} | ${formatBytesDecimal(state.files[0].size)}`;
    $("btnReset").disabled = false;

    // Show preview of original immediately?
    // User wants "Slider Comparison" after processing.
    // Maybe show just the original first?
    // For now, let's keep the prompt but update text.
    const promptText = document.querySelector(".upload-prompt p");
    if (promptText) promptText.textContent = `${files[0].name} 已加载`;
  }
}

// Bindings
function bindUI() {
  const grainSlider = $("grainStrength");
  grainSlider.addEventListener("input", (e) => {
    $("grainStrengthVal").textContent = fmt(e.target.value);
  });

  $("btnProcess").addEventListener("click", processImage);
  $("btnReset").addEventListener("click", resetSelection);

  initProgressUI();
  initComparisonSlider();
  initDragDrop();
}

async function main() {
  bindUI();
  setStatus("系统初始化...");

  // Add fade-in to main elements
  document.querySelector('.viewport').classList.add('fade-in');
  document.querySelector('.controls').classList.add('fade-in');

  try {
    const presets = await loadPresets();
    state.presets = presets;
    const preferred = presets.default_film_type;
    if (preferred && presets.film_types.includes(preferred)) {
      state.currentFilm = preferred;
    } else if (presets.film_types.length) {
      state.currentFilm = presets.film_types[0];
    }
    renderFilmShelf(presets);
    setStatus("系统就绪");
  } catch (e) {
    setStatus("系统错误", true);
    console.error(e);
  }
}

main();
