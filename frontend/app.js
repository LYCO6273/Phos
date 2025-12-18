const state = {
  // grainEnabled: true, // Removed
  files: [],
  presets: null,
  currentFilm: "NC200",
  processedBlobUrl: null,
  originalBlobUrl: null,
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

function fmt(n) {
  const v = Number(n);
  if (Number.isNaN(v)) return "";
  return v.toFixed(2).replace(/\.00$/, "");
}

// --- API ---
async function loadPresets() {
  // Rich mock data
  return {
    film_types: ["GOLD200", "PORTRA400", "FUJI200", "ILFORD5", "CINE800", "EKTAR100", "TRI_X400", "NATURA1600"],
    meta: {
      "GOLD200": {
        name: "Kodak Gold 200",
        type: "彩色负片",
        iso: 200,
        weather: "晴天 / 黄金时刻",
        scene: "街头, 旅行, 怀旧",
        desc: "经典的暖色调，带有复古的金色光泽。记忆的完美载体。",
        brandColor: "#FFB800"
      },
      "PORTRA400": {
        name: "Kodak Portra 400",
        type: "专业彩色负片",
        iso: 400,
        weather: "阴天 / 室内",
        scene: "人像, 婚礼, 肤色",
        desc: "世界上颗粒最细腻的胶卷。卓越的肤色还原和宽容度。",
        brandColor: "#E6C200"
      },
      "FUJI200": {
        name: "Fujifilm C200",
        type: "彩色负片",
        iso: 200,
        weather: "日光 / 自然",
        scene: "风景, 绿植, 城市",
        desc: "色调偏冷，绿色和洋红色表现鲜艳。清晰锐利。",
        brandColor: "#48A14D"
      },
      "ILFORD5": {
        name: "Ilford HP5 Plus",
        type: "黑白负片",
        iso: 400,
        weather: "全天候 / 暗光",
        scene: "纪实, 街头, 艺术",
        desc: "颗粒细腻，边缘对比度好。新闻摄影师的首选。",
        brandColor: "#FFFFFF"
      },
      "CINE800": {
        name: "CineStill 800T",
        type: "电影灯光片",
        iso: 800,
        weather: "夜晚 / 人造光",
        scene: "夜景, 霓虹灯, 电影感",
        desc: "用于静态摄影的电影胶卷。以灯光周围的红色光晕闻名。",
        brandColor: "#D12E2E"
      },
      "EKTAR100": {
        name: "Kodak Ektar 100",
        type: "专业彩色负片",
        iso: 100,
        weather: "晴天 / 户外",
        scene: "风景, 建筑, 产品",
        desc: "世界上颗粒最细的彩色负片。色彩极其鲜艳，反差高。",
        brandColor: "#0095DA"
      },
      "TRI_X400": {
        name: "Kodak Tri-X 400",
        type: "黑白负片",
        iso: 400,
        weather: "全天候 / 街头",
        scene: "新闻, 纪实, 人文",
        desc: "经典的黑白胶卷，颗粒感明显，对比度强烈，充满戏剧性。",
        brandColor: "#FFD700"
      },
      "NATURA1600": {
        name: "Fujifilm Natura 1600",
        type: "高速彩色负片",
        iso: 1600,
        weather: "暗光 / 室内",
        scene: "夜景, 抓拍, 氛围",
        desc: "月光机专用卷。在极低光照下也能保持自然的色彩还原。",
        brandColor: "#E5004F"
      }
    }
  };
}

// ... (readOptions remains same)

// Film Selector with Carousel
let currentIndex = 0;
let carouselOffset = 0;
let isDragging = false;
let startX = 0;
let currentX = 0;

function renderFilmShelf(presets) {
  const track = $("carouselTrack");
  track.innerHTML = "";

  const filmImages = [
    "assets/film-cans/segment_01-removebg-preview.png",
    "assets/film-cans/segment_02-removebg-preview.png",
    "assets/film-cans/segment_03-removebg-preview.png",
    "assets/film-cans/segment_04-removebg-preview.png",
    "assets/film-cans/segment_05-removebg-preview.png",
    "assets/film-cans/segment_06-removebg-preview.png",
    "assets/film-cans/segment_07-removebg-preview.png",
    "assets/film-cans/segment_08-removebg-preview.png"
  ];

  presets.film_types.forEach((type, index) => {
    const info = presets.meta[type];
    const el = document.createElement("div");
    el.className = `film-canister ${type === state.currentFilm ? "active selected" : ""}`;
    el.dataset.film = type;
    el.dataset.index = index;

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
      <img src="${filmImages[index]}" alt="${info.name}" class="film-canister-img">
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

  $("infoType").textContent = info.type;
  $("infoIso").textContent = info.iso;
  $("infoRec").textContent = info.weather;
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

  // Add loading state to button
  const originalBtnText = $("btnProcess").textContent;
  $("btnProcess").textContent = "处理中...";

  try {
    const options = readOptions();
    const fd = new FormData();
    fd.append("file", file);
    Object.entries(options).forEach(([k, v]) => fd.append(k, String(v)));

    const res = await fetch("/api/process", { method: "POST", body: fd });
    if (!res.ok) throw new Error("冲洗失败");

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

    $("imageMeta").textContent = `${file.name} | ${(file.size / 1024 / 1024).toFixed(2)}MB`;

    $("btnDownload").disabled = false;
    $("btnDownload").onclick = () => {
      const a = document.createElement("a");
      a.href = url;
      a.download = `phos_${state.currentFilm}_${file.name}`;
      a.click();
    };

    setStatus("冲洗完成");

  } catch (e) {
    setStatus(`错误: ${e.message}`, true);
  } finally {
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
    state.files = Array.from(files);
    setStatus("胶卷已加载");
    $("imageMeta").textContent = "胶卷就绪，准备冲洗";

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
    renderFilmShelf(presets);
    setStatus("系统就绪");
  } catch (e) {
    setStatus("系统错误", true);
    console.error(e);
  }
}

main();
