/* ============================================================
   penDNA — frontend
   ============================================================ */
(() => {
'use strict';

// ------------------------------------------------------------
// constants
// ------------------------------------------------------------
const COLORS = {
  bg:      '#0D0D0F',
  surface: '#161619',
  border:  '#2A2A2E',
  text:    '#E8E6E1',
  muted:   '#7A7A7D',
  cyan:    '#00E5CC',
  coral:   '#FF5E3A',
  amber:   '#FFAA2A',
  violet:  '#8B5CF6',
};
const CORAL_SHADES = ['#FF5E3A', '#FF8A5A', '#E04020', '#FFB088'];
const TARGET_FPS = 30;

// ------------------------------------------------------------
// DOM
// ------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const els = {
  // top bar
  penState: $('pen-state'), dotPen: $('dot-pen'),
  camState: $('cam-state'), dotCam: $('dot-cam'),
  latValue: $('lat-value'),
  capState: $('cap-state'), dotCap: $('dot-cap'),
  profCount: $('prof-count'),
  // left
  webcam: $('webcam'),
  pathCanvas: $('pathCanvas'),
  bars: document.querySelectorAll('#pressureBars .bar-row'),
  imu: {
    ax: $('imu-ax'), ay: $('imu-ay'), az: $('imu-az'),
    gx: $('imu-gx'), gy: $('imu-gy'), gz: $('imu-gz'),
  },
  penStatus: $('pen-status'),
  btnCapture: $('btn-capture'),
  btnClear: $('btn-clear'),
  // center
  prompt: $('prompt'),
  btnGenerate: $('btn-generate'),
  splitToggle: $('split-toggle'),
  styleTag: $('style-tag'),
  drawCanvas: $('drawCanvas'),
  drawCanvasB: $('drawCanvasB'),
  drawStack: $('drawStack'),
  canvasOverlay: $('canvas-overlay'),
  overlayText: $('overlay-text'),
  splitDivider: $('split-divider'),
  splitLabels: $('split-labels'),
  // right
  tremorChart: $('tremorChart'),
  gripChart: $('gripChart'),
  velocityChart: $('velocityChart'),
  radarChart: $('radarChart'),
  signatureCanvas: $('signatureCanvas'),
  tremorPeak: $('tremor-peak'),
  velCurrent: $('vel-current'),
  sigHash: $('sig-hash'),
};

// ------------------------------------------------------------
// state
// ------------------------------------------------------------
const state = {
  capturing: false,
  generating: false,
  splitView: false,
  pathPoints: [],         // cyan path overlay points
  pressure: [0,0,0,0],
  imu: {ax:0, ay:0, az:0, gx:0, gy:0, gz:0},
  penDown: false,
  // chart history
  pressureHistory: [[],[],[],[]],
  velocityHistory: [],
  tremorSpectrum: new Array(40).fill(0),
  gripAvg: [0.5, 0.5, 0.5, 0.5],
  styleVector: null,
  currentUser: 'USER_A',
  strokeQueue: [],        // queued strokes for animation
  activeStroke: null,
  activeCanvas: 'A',      // which canvas to draw onto next
};

// ------------------------------------------------------------
// chart.js defaults
// ------------------------------------------------------------
Chart.defaults.color = COLORS.muted;
Chart.defaults.borderColor = COLORS.border;
Chart.defaults.font.family = '"JetBrains Mono", "SF Mono", monospace';
Chart.defaults.font.size = 9;
Chart.defaults.animation = false;
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;

const baseScale = (max, min = 0) => ({
  min, max,
  grid: { color: 'rgba(42,42,46,0.6)', drawTicks: false },
  border: { color: COLORS.border, display: false },
  ticks: {
    color: COLORS.muted,
    font: { family: '"JetBrains Mono"', size: 8 },
    padding: 2, maxTicksLimit: 4,
  },
});

// ------------------------------------------------------------
// charts
// ------------------------------------------------------------
const tremorLabels = Array.from({length: 40}, (_, i) => (i * 0.5).toFixed(1));
const tremorChart = new Chart(els.tremorChart, {
  type: 'line',
  data: {
    labels: tremorLabels,
    datasets: [{
      data: state.tremorSpectrum,
      borderColor: COLORS.cyan,
      borderWidth: 1.4,
      backgroundColor: 'rgba(0,229,204,0.10)',
      fill: true,
      pointRadius: 0,
      tension: 0.35,
    }],
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { ...baseScale(40), title: { display: false },
           ticks: { ...baseScale(40).ticks, callback: (v, i) => (i % 8 === 0) ? `${tremorLabels[i]}Hz` : '' } },
      y: { ...baseScale(1), ticks: { ...baseScale(1).ticks, callback: () => '' } },
    },
  },
});

const gripChart = new Chart(els.gripChart, {
  type: 'line',
  data: {
    labels: Array.from({length: 60}, (_, i) => i),
    datasets: [0,1,2,3].map(i => ({
      data: new Array(60).fill(0),
      borderColor: CORAL_SHADES[i],
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    })),
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { ...baseScale(60), ticks: { display: false }, grid: { display: false } },
      y: { ...baseScale(1), ticks: { ...baseScale(1).ticks, callback: () => '' } },
    },
  },
});

const velocityChart = new Chart(els.velocityChart, {
  type: 'line',
  data: {
    labels: Array.from({length: 80}, (_, i) => i),
    datasets: [{
      data: new Array(80).fill(0),
      borderColor: COLORS.amber,
      borderWidth: 1.4,
      backgroundColor: 'rgba(255,170,42,0.18)',
      fill: true,
      pointRadius: 0,
      tension: 0.4,
    }],
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { ...baseScale(80), ticks: { display: false }, grid: { display: false } },
      y: { ...baseScale(300), ticks: { ...baseScale(300).ticks,
        callback: (v) => (v % 100 === 0) ? v : '' } },
    },
  },
});

const radarChart = new Chart(els.radarChart, {
  type: 'radar',
  data: {
    labels: ['S1','S2','S3','S4'],
    datasets: [{
      data: [0.5, 0.5, 0.5, 0.5],
      borderColor: COLORS.coral,
      backgroundColor: 'rgba(255,94,58,0.22)',
      borderWidth: 1.4,
      pointRadius: 2,
      pointBackgroundColor: COLORS.coral,
    }],
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      r: {
        min: 0, max: 1,
        grid:  { color: 'rgba(42,42,46,0.7)' },
        angleLines: { color: 'rgba(42,42,46,0.7)' },
        pointLabels: { color: COLORS.muted, font: { family: '"JetBrains Mono"', size: 9 } },
        ticks: { display: false, count: 3 },
      },
    },
  },
});

// ------------------------------------------------------------
// signature canvas — lissajous from style vector
// ------------------------------------------------------------
const sigCtx = els.signatureCanvas.getContext('2d');
function drawSignature(vec) {
  const c = els.signatureCanvas;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * devicePixelRatio;
  c.height = h * devicePixelRatio;
  sigCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  sigCtx.clearRect(0, 0, w, h);

  // params from style vector or defaults
  const v = vec || state.styleVector || defaultStyleVec();
  const cx = w / 2, cy = h / 2;
  const r = Math.min(w, h) * 0.42;

  // lissajous: x = sin(a*t + phi), y = sin(b*t)
  const a = 2 + (v[0] % 6);
  const b = 3 + (v[1] % 6);
  const phi = v[2] * Math.PI;
  const wobble = 0.05 + v[3] * 0.15;

  sigCtx.strokeStyle = COLORS.violet;
  sigCtx.lineWidth = 1.1;
  sigCtx.shadowColor = 'rgba(139,92,246,0.55)';
  sigCtx.shadowBlur = 6;

  sigCtx.beginPath();
  const STEPS = 600;
  for (let i = 0; i <= STEPS; i++) {
    const t = (i / STEPS) * Math.PI * 2;
    const wob = 1 + wobble * Math.sin(7 * t + phi);
    const x = cx + Math.sin(a * t + phi) * r * wob;
    const y = cy + Math.sin(b * t) * r * wob;
    if (i === 0) sigCtx.moveTo(x, y);
    else sigCtx.lineTo(x, y);
  }
  sigCtx.stroke();
  sigCtx.shadowBlur = 0;

  // hash text
  const hash = v.slice(0, 4).map(n => Math.floor(n * 255).toString(16).padStart(2, '0')).join('');
  els.sigHash.textContent = hash.toUpperCase();
}

function defaultStyleVec() {
  // pseudo-random per-session vector so it looks personal even when idle
  if (!state._sessionVec) {
    state._sessionVec = Array.from({length: 8}, () => Math.random());
  }
  return state._sessionVec;
}

// ------------------------------------------------------------
// pen path overlay on webcam
// ------------------------------------------------------------
const pathCtx = els.pathCanvas.getContext('2d');
function resizePathCanvas() {
  const c = els.pathCanvas;
  const r = c.getBoundingClientRect();
  c.width = r.width * devicePixelRatio;
  c.height = r.height * devicePixelRatio;
  pathCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
window.addEventListener('resize', () => {
  resizePathCanvas();
  resizeDrawCanvases();
  drawSignature();
});

function pushPathPoint(p) {
  // p = {x: 0..1 normalized, y: 0..1 normalized, t: timestamp ms}
  state.pathPoints.push({ ...p, t: p.t || performance.now() });
  // cap at ~6s of history
  const cutoff = performance.now() - 6000;
  while (state.pathPoints.length && state.pathPoints[0].t < cutoff) {
    state.pathPoints.shift();
  }
}

function renderPath() {
  const c = els.pathCanvas;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  pathCtx.clearRect(0, 0, w, h);

  const pts = state.pathPoints;
  if (pts.length < 2) return;
  const now = performance.now();

  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (b.lift || a.lift) continue;
    const age = (now - b.t) / 6000;
    const alpha = Math.max(0, 1 - age);
    pathCtx.strokeStyle = `rgba(0,229,204,${alpha.toFixed(3)})`;
    pathCtx.lineWidth = 1.6;
    pathCtx.shadowColor = 'rgba(0,229,204,0.6)';
    pathCtx.shadowBlur = 4;
    pathCtx.beginPath();
    pathCtx.moveTo(a.x * w, a.y * h);
    pathCtx.lineTo(b.x * w, b.y * h);
    pathCtx.stroke();
  }
  pathCtx.shadowBlur = 0;
}

// ------------------------------------------------------------
// generation canvas
// ------------------------------------------------------------
const drawCtx = els.drawCanvas.getContext('2d');
const drawCtxB = els.drawCanvasB.getContext('2d');

function resizeDrawCanvases() {
  for (const c of [els.drawCanvas, els.drawCanvasB]) {
    const r = c.getBoundingClientRect();
    if (!r.width) continue;
    c.width = r.width * devicePixelRatio;
    c.height = r.height * devicePixelRatio;
    const ctx = c.getContext('2d');
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
}

function clearDrawCanvas(which = 'both') {
  if (which === 'A' || which === 'both') {
    drawCtx.clearRect(0, 0, els.drawCanvas.clientWidth, els.drawCanvas.clientHeight);
  }
  if (which === 'B' || which === 'both') {
    drawCtxB.clearRect(0, 0, els.drawCanvasB.clientWidth, els.drawCanvasB.clientHeight);
  }
}

function showOverlay(text, klass) {
  els.canvasOverlay.classList.remove('hidden');
  els.overlayText.textContent = text;
  els.overlayText.className = 'overlay-text' + (klass ? ' ' + klass : '');
}
function hideOverlay() {
  els.canvasOverlay.classList.add('hidden');
}

// queue a stroke for animation
// stroke = { points: [{x, y, pressure, t}], color, thickness, jitter, speedMs, canvas: 'A'|'B' }
function enqueueStroke(stroke) {
  state.strokeQueue.push(stroke);
  hideOverlay();
}

function getDrawCtxFor(which) {
  const c = which === 'B' ? els.drawCanvasB : els.drawCanvas;
  const ctx = which === 'B' ? drawCtxB : drawCtx;
  return { c, ctx };
}

function tickStrokeAnimation(now) {
  // start next stroke if needed
  if (!state.activeStroke && state.strokeQueue.length) {
    state.activeStroke = state.strokeQueue.shift();
    state.activeStroke.startT = now;
    state.activeStroke.lastIdx = 0;
  }
  const s = state.activeStroke;
  if (!s) return;

  const { c, ctx } = getDrawCtxFor(s.canvas || 'A');
  const w = c.clientWidth, h = c.clientHeight;
  const pts = s.points;
  if (!pts.length) { state.activeStroke = null; return; }

  const elapsed = now - s.startT;
  const targetIdx = Math.min(pts.length - 1,
    Math.floor((elapsed / (s.speedMs || 1000)) * pts.length));

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = s.color || COLORS.text;
  ctx.shadowColor = s.glow || 'rgba(232,230,225,0.0)';
  ctx.shadowBlur = s.glow ? 4 : 0;

  for (let i = s.lastIdx; i < targetIdx; i++) {
    const a = pts[i], b = pts[i + 1];
    if (!b) break;
    const press = (a.pressure ?? 0.5);
    const lw = (s.thickness || 1.5) * (0.4 + press * 1.4);
    const jit = s.jitter || 0;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(a.x * w + (Math.random()-0.5) * jit, a.y * h + (Math.random()-0.5) * jit);
    ctx.lineTo(b.x * w + (Math.random()-0.5) * jit, b.y * h + (Math.random()-0.5) * jit);
    ctx.stroke();
  }
  s.lastIdx = targetIdx;

  if (targetIdx >= pts.length - 1) {
    state.activeStroke = null;
  }
}

// ------------------------------------------------------------
// data update helpers (with flash)
// ------------------------------------------------------------
function flashEl(el, color) {
  el.classList.add('flash');
  if (color) el.style.color = color;
  clearTimeout(el._flash);
  el._flash = setTimeout(() => {
    el.classList.remove('flash');
    el.style.color = '';
  }, 120);
}

function updatePressureBars(values) {
  state.pressure = values;
  els.bars.forEach((row, i) => {
    const v = Math.max(0, Math.min(1, values[i] || 0));
    row.querySelector('.bar-fill').style.width = (v * 100).toFixed(1) + '%';
    const valEl = row.querySelector('.bar-value');
    valEl.textContent = v.toFixed(2);
    flashEl(valEl);
    state.pressureHistory[i].push(v);
    if (state.pressureHistory[i].length > 60) state.pressureHistory[i].shift();
  });
}

function updateIMU(d) {
  state.imu = d;
  els.imu.ax.textContent = d.ax.toFixed(1); flashEl(els.imu.ax);
  els.imu.ay.textContent = d.ay.toFixed(1); flashEl(els.imu.ay);
  els.imu.az.textContent = d.az.toFixed(1); flashEl(els.imu.az);
  els.imu.gx.textContent = d.gx.toFixed(1); flashEl(els.imu.gx);
  els.imu.gy.textContent = d.gy.toFixed(1); flashEl(els.imu.gy);
  els.imu.gz.textContent = d.gz.toFixed(1); flashEl(els.imu.gz);
}

function updatePenStatus(down) {
  state.penDown = down;
  els.penStatus.textContent = down ? 'PEN DOWN' : 'PEN UP';
  els.penStatus.className = 'pen-state-value ' + (down ? 'pen-down' : 'pen-up');
}

function updateStyleVector(vec) {
  state.styleVector = vec;
  drawSignature(vec);
  // update grip avg into radar
  if (vec.grip && vec.grip.length === 4) {
    radarChart.data.datasets[0].data = vec.grip;
    radarChart.update('none');
  }
  if (vec.tremor && vec.tremor.length) {
    state.tremorSpectrum = vec.tremor;
    tremorChart.data.datasets[0].data = vec.tremor;
    // peak
    let pi = 0, pv = 0;
    for (let i = 0; i < vec.tremor.length; i++) {
      if (vec.tremor[i] > pv) { pv = vec.tremor[i]; pi = i; }
    }
    els.tremorPeak.textContent = (pi * 0.5).toFixed(1);
    tremorChart.update('none');
  }
}

// ------------------------------------------------------------
// 30fps render loop
// ------------------------------------------------------------
let lastFrame = 0;
function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  if (now - lastFrame < 1000 / TARGET_FPS) return;
  lastFrame = now;

  renderPath();
  tickStrokeAnimation(now);

  // push grip history into chart
  for (let i = 0; i < 4; i++) {
    gripChart.data.datasets[i].data = state.pressureHistory[i].slice(-60);
  }
  gripChart.update('none');

  // velocity chart
  velocityChart.data.datasets[0].data = state.velocityHistory.slice(-80);
  velocityChart.update('none');
}
requestAnimationFrame(renderLoop);

// ------------------------------------------------------------
// webcam
// ------------------------------------------------------------
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }, audio: false,
    });
    els.webcam.srcObject = stream;
    els.camState.textContent = 'ACTIVE';
    els.dotCam.className = 'dot dot-cyan';
  } catch (err) {
    els.camState.textContent = 'OFFLINE';
    els.dotCam.className = 'dot dot-coral';
    console.warn('webcam unavailable:', err.message);
  }
  resizePathCanvas();
}

// ------------------------------------------------------------
// socket.io
// ------------------------------------------------------------
let socket = null;
function initSocket() {
  try {
    socket = io({ transports: ['websocket', 'polling'], reconnection: true });
  } catch (e) {
    console.warn('socket init failed', e);
    return;
  }

  socket.on('connect', () => {
    els.penState.textContent = 'CONNECTED';
    els.dotPen.className = 'dot dot-cyan';
  });
  socket.on('disconnect', () => {
    els.penState.textContent = 'OFFLINE';
    els.dotPen.className = 'dot dot-coral';
  });
  socket.on('connect_error', () => {
    els.penState.textContent = 'OFFLINE';
    els.dotPen.className = 'dot dot-coral';
  });

  socket.on('sensor_data', (d) => {
    if (d.pressure) updatePressureBars(d.pressure);
    if (d.imu) updateIMU(d.imu);
    if (typeof d.pen_down === 'boolean') updatePenStatus(d.pen_down);
    if (typeof d.latency === 'number') els.latValue.textContent = d.latency.toFixed(0);
    if (typeof d.velocity === 'number') {
      state.velocityHistory.push(d.velocity);
      if (state.velocityHistory.length > 200) state.velocityHistory.shift();
      els.velCurrent.textContent = d.velocity.toFixed(0);
    }
  });

  socket.on('path_point', (p) => {
    pushPathPoint(p);
  });

  socket.on('style_update', (vec) => {
    updateStyleVector(vec);
    if (vec.user) {
      els.styleTag.textContent = vec.user;
      state.currentUser = vec.user;
    }
    if (typeof vec.profiles === 'number') els.profCount.textContent = vec.profiles;
  });

  socket.on('generation_start', (d) => {
    state.generating = true;
    els.btnGenerate.classList.add('generating');
    showOverlay('GENERATING…', 'generating');
    const which = (d && d.canvas) || (state.splitView ? state.activeCanvas : 'A');
    clearDrawCanvas(which);
  });

  socket.on('stroke_data', (stroke) => {
    enqueueStroke({
      points: stroke.points || [],
      color: stroke.color || COLORS.text,
      thickness: stroke.thickness ?? 1.5,
      jitter: stroke.jitter ?? 0,
      speedMs: stroke.speed_ms ?? stroke.speedMs ?? 800,
      glow: stroke.glow,
      canvas: stroke.canvas || (state.splitView ? state.activeCanvas : 'A'),
    });
  });

  socket.on('generation_done', () => {
    state.generating = false;
    els.btnGenerate.classList.remove('generating');
    if (state.splitView) state.activeCanvas = state.activeCanvas === 'A' ? 'B' : 'A';
  });
}

// ------------------------------------------------------------
// UI events
// ------------------------------------------------------------
els.btnCapture.addEventListener('click', () => {
  state.capturing = !state.capturing;
  if (state.capturing) {
    els.btnCapture.textContent = 'STOP CAPTURE';
    els.btnCapture.classList.add('recording');
    els.capState.textContent = 'RECORDING';
    els.capState.classList.add('pulse');
    els.dotCap.className = 'dot dot-cyan';
    socket && socket.emit('start_capture');
  } else {
    els.btnCapture.textContent = 'START CAPTURE';
    els.btnCapture.classList.remove('recording');
    els.capState.textContent = 'IDLE';
    els.capState.classList.remove('pulse');
    els.dotCap.className = 'dot dot-gray';
    socket && socket.emit('stop_capture');
  }
});

els.btnClear.addEventListener('click', () => {
  state.pathPoints = [];
  state.strokeQueue = [];
  state.activeStroke = null;
  clearDrawCanvas('both');
  showOverlay('draw something first, then generate');
  socket && socket.emit('clear');
});

els.btnGenerate.addEventListener('click', () => {
  const prompt = els.prompt.value.trim();
  if (!prompt) { els.prompt.focus(); return; }
  if (state.generating) return;
  socket && socket.emit('generate', { prompt, user: state.currentUser, split: state.splitView });
  // optimistic UI
  showOverlay('GENERATING…', 'generating');
  els.btnGenerate.classList.add('generating');
});

els.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') els.btnGenerate.click();
});

els.splitToggle.addEventListener('change', (e) => {
  state.splitView = e.target.checked;
  if (state.splitView) {
    els.drawCanvasB.classList.remove('hidden');
    els.splitDivider.classList.remove('hidden');
    els.splitLabels.classList.remove('hidden');
    // shrink each canvas to half width visually
    els.drawCanvas.style.width = '50%';
    els.drawCanvasB.style.left = '50%';
    els.drawCanvasB.style.width = '50%';
    resizeDrawCanvases();
  } else {
    els.drawCanvasB.classList.add('hidden');
    els.splitDivider.classList.add('hidden');
    els.splitLabels.classList.add('hidden');
    els.drawCanvas.style.width = '100%';
    els.drawCanvasB.style.width = '';
    els.drawCanvasB.style.left = '';
    resizeDrawCanvases();
  }
});

// ------------------------------------------------------------
// boot
// ------------------------------------------------------------
function boot() {
  resizePathCanvas();
  resizeDrawCanvases();
  drawSignature();
  showOverlay('draw something first, then generate');
  initWebcam();
  initSocket();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

})();
