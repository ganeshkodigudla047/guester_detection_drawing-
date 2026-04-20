/* ============================================================
   Hand Gesture Drawing — Browser JS
   MediaPipe Hands + Canvas API
   Gestures:
     Index only          → DRAW
     Index + Middle      → CURSOR
     Full palm           → ERASE
     Pinch tap           → UNDO
     Pinch double-tap    → REDO
     Pinch hold >0.5s    → MOVE (drag canvas)
     Both hands pinch    → ZOOM
   ============================================================ */

'use strict';

// ── Constants ────────────────────────────────────────────────
const PINCH_THRESH    = 0.06;   // normalised distance (0-1 of frame width)
const HOLD_SECS       = 0.5;
const TAP_MAX_SECS    = 0.30;
const DOUBLE_SECS     = 0.40;
const COOLDOWN_SECS   = 0.50;
const BANNER_SECS     = 1.2;
const SMOOTH_ALPHA    = 0.45;   // EMA factor for cursor (lower = smoother)
const MIN_DRAW_PX     = 4;
const ERASER_R        = 40;
const UNDO_MAX        = 30;
const DEBOUNCE_FRAMES = 3;

// ── DOM refs ─────────────────────────────────────────────────
const video         = document.getElementById('webcam');
const drawCanvas    = document.getElementById('draw-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const dctx          = drawCanvas.getContext('2d');
const octx          = overlayCanvas.getContext('2d');
const modeLabel   = document.getElementById('mode-label');
const fpsLabel    = document.getElementById('fps-label');
const statusMsg   = document.getElementById('status-msg');
const banner      = document.getElementById('action-banner');

// ── State ────────────────────────────────────────────────────
let color     = '#ff0000';
let brushSize = 6;
let prevPt    = null;       // {x,y} last drawn point
let smoothPt  = null;       // {x,y} EMA cursor

// Undo / Redo stacks — store ImageData snapshots
const undoStack = [];
const redoStack = [];

// Pinch tap/hold detector state
let pinchActive    = false;
let pinchStart     = 0;
let tapCount       = 0;
let lastTapEnd     = 0;
let lastAction     = 0;
let isMoveActive   = false;

// Move (drag) state
let moveAnchor     = null;   // {x,y} pinch midpoint when drag started
let moveOffsetBase = {x:0, y:0};
let canvasOffset   = {x:0, y:0};  // current translation of draw canvas

// Zoom state
let inZoom         = false;
let zoomInitDist   = 0;
let zoomInitScale  = 1;
let zoomScale      = 1;

// Gesture debounce
let modeCandidate  = 'IDLE';
let modeStreak     = 0;
let activeMode     = 'IDLE';

// Shape detection
let strokePoints   = [];    // points collected during current DRAW stroke
let wasDrawing     = false;

// FPS
let lastFrameTime  = performance.now();
let frameCount     = 0;
let fps            = 0;

// Banner timer
let bannerTimer    = null;

// ── Toolbar wiring ───────────────────────────────────────────
document.querySelectorAll('.color-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    color = btn.dataset.color;
  });
});

document.querySelectorAll('.brush-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.brush-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    brushSize = parseInt(btn.dataset.size);
  });
});

document.getElementById('btn-undo').addEventListener('click', doUndo);
document.getElementById('btn-redo').addEventListener('click', doRedo);
document.getElementById('btn-clear').addEventListener('click', doClear);
document.getElementById('btn-save').addEventListener('click', doSave);

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'z') doUndo();
  if (e.ctrlKey && (e.key === 'y' || e.key === 'Z')) doRedo();
});

// ── Canvas resize — match pixel dims to CSS layout size ──────
function syncCanvasSize() {
  const wrap = document.getElementById('canvas-wrap');
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (w < 1 || h < 1) return;
  [drawCanvas, overlayCanvas].forEach(c => {
    if (c.width !== w || c.height !== h) {
      // preserve drawing content
      let saved = null;
      if (c === drawCanvas && c.width > 0 && c.height > 0) {
        saved = dctx.getImageData(0, 0, c.width, c.height);
      }
      c.width  = w;
      c.height = h;
      if (saved) dctx.putImageData(saved, 0, 0);
    }
  });
}

// Keep canvases sized correctly whenever the window resizes
const ro = new ResizeObserver(syncCanvasSize);
ro.observe(document.getElementById('canvas-wrap'));
window.addEventListener('resize', syncCanvasSize);

// ── Undo / Redo ──────────────────────────────────────────────
function snapshot() {
  const data = dctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  undoStack.push(data);
  if (undoStack.length > UNDO_MAX) undoStack.shift();
  redoStack.length = 0;
}

function doUndo() {
  if (!undoStack.length) return;
  redoStack.push(dctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
  dctx.putImageData(undoStack.pop(), 0, 0);
  showBanner('UNDO', 'undo');
}

function doRedo() {
  if (!redoStack.length) return;
  undoStack.push(dctx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
  dctx.putImageData(redoStack.pop(), 0, 0);
  showBanner('REDO', 'redo');
}

function doClear() {
  snapshot();
  dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  canvasOffset = {x:0, y:0};
  zoomScale = 1;
  applyCanvasTransform();
}

function doSave() {
  const link = document.createElement('a');
  link.download = `drawing_${Date.now()}.png`;
  link.href = drawCanvas.toDataURL('image/png');
  link.click();
}

// ── Banner ───────────────────────────────────────────────────
function showBanner(text, cls) {
  banner.textContent = text;
  banner.className   = `show ${cls}`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { banner.className = ''; }, BANNER_SECS * 1000);
}

// ── Canvas transform (move + zoom) ───────────────────────────
function applyCanvasTransform() {
  drawCanvas.style.transform =
    `translate(${canvasOffset.x}px, ${canvasOffset.y}px) scale(${zoomScale})`;
  drawCanvas.style.transformOrigin = 'center center';
}

// ── Helpers ──────────────────────────────────────────────────
function lm(landmarks, idx) {
  // Canvas pixel coords — video is CSS-mirrored so we flip X here too
  const W = overlayCanvas.width;
  const H = overlayCanvas.height;
  return {
    x: (1 - landmarks[idx].x) * W,
    y: landmarks[idx].y * H
  };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isPinching(landmarks) {
  const thumb = lm(landmarks, 4);
  const index = lm(landmarks, 8);
  const d = dist(thumb, index) / overlayCanvas.width;
  return d < PINCH_THRESH;
}

function fingerUp(landmarks, tipId, pipId) {
  return landmarks[tipId].y < landmarks[pipId].y;
}

// ── Gesture debouncer ────────────────────────────────────────
function debounceMode(raw) {
  if (raw === modeCandidate) {
    modeStreak++;
  } else {
    modeCandidate = raw;
    modeStreak    = 1;
  }
  if (modeStreak >= DEBOUNCE_FRAMES) activeMode = modeCandidate;
  return activeMode;
}

// ── Detect single-hand gesture ───────────────────────────────
function detectGesture(landmarks) {
  const thumb  = fingerUp(landmarks, 4, 3);
  const index  = fingerUp(landmarks, 8, 6);
  const middle = fingerUp(landmarks, 12, 10);
  const ring   = fingerUp(landmarks, 16, 14);
  const pinky  = fingerUp(landmarks, 20, 18);

  // Full palm → ERASE
  if (thumb && index && middle && ring && pinky) return 'ERASE';

  // Pinch → MOVE / UNDO / REDO (handled by pinch detector)
  if (isPinching(landmarks) && !middle && !ring && !pinky) return 'MOVE';

  // Index only → DRAW
  if (index && !thumb && !middle && !ring && !pinky) return 'DRAW';

  // Index + middle → CURSOR
  if (index && middle && !ring && !pinky) return 'CURSOR';

  return 'IDLE';
}

// ── Pinch tap/hold detector ──────────────────────────────────
// Returns 'UNDO', 'REDO', 'MOVE_START', or null
function updatePinchDetector(pinching) {
  const now = performance.now() / 1000;
  let action = null;

  if (pinching && !pinchActive) {
    pinchActive = true;
    pinchStart  = now;
  } else if (pinching && pinchActive) {
    if ((now - pinchStart) >= HOLD_SECS && !isMoveActive) {
      isMoveActive = true;
      action = 'MOVE_START';
      tapCount = 0;
      showBanner('MOVE MODE', 'move');
    }
  } else if (!pinching && pinchActive) {
    const dur = now - pinchStart;
    pinchActive = false;
    if (isMoveActive) {
      isMoveActive = false;
      moveAnchor   = null;
    } else if (dur < TAP_MAX_SECS) {
      if (tapCount === 1 && (now - lastTapEnd) < DOUBLE_SECS) {
        if (now - lastAction >= COOLDOWN_SECS) {
          action = 'REDO';
          lastAction = now;
          doRedo();
        }
        tapCount = 0;
      } else {
        tapCount   = 1;
        lastTapEnd = now;
      }
    }
  }

  // Single-tap window expired → UNDO
  if (!pinching && tapCount === 1 && (performance.now()/1000 - lastTapEnd) >= DOUBLE_SECS) {
    if (performance.now()/1000 - lastAction >= COOLDOWN_SECS) {
      action = 'UNDO';
      lastAction = performance.now()/1000;
      doUndo();
    }
    tapCount = 0;
  }

  return action;
}

// ── Shape detection ──────────────────────────────────────────
function detectShape(pts) {
  if (pts.length < 20) return null;

  // Smooth points
  const smoothed = pts.map((p, i) => {
    const lo = Math.max(0, i - 2);
    const hi = Math.min(pts.length - 1, i + 2);
    let sx = 0, sy = 0, n = 0;
    for (let j = lo; j <= hi; j++) { sx += pts[j].x; sy += pts[j].y; n++; }
    return {x: sx/n, y: sy/n};
  });

  // Bounding box
  const xs = smoothed.map(p => p.x);
  const ys = smoothed.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = maxX - minX, h = maxY - minY;
  if (w * h < 1000) return null;

  // Perimeter
  let peri = 0;
  for (let i = 1; i < smoothed.length; i++)
    peri += dist(smoothed[i-1], smoothed[i]);

  // Douglas-Peucker simplification
  const eps = 0.03 * peri;
  const approx = douglasPeucker(smoothed, eps);
  const n = approx.length;

  let shape = null;
  if (n <= 2)      shape = 'LINE';
  else if (n === 3) shape = 'TRIANGLE';
  else if (n === 4) {
    const aspect = w / h;
    shape = (aspect > 0.85 && aspect < 1.15) ? 'SQUARE' : 'RECTANGLE';
  } else if (n > 6) {
    // Circularity
    const area = polygonArea(smoothed);
    const r    = Math.max(w, h) / 2;
    const circ = area / (Math.PI * r * r);
    shape = circ > 0.60 ? 'CIRCLE' : 'POLYGON';
  } else {
    shape = 'POLYGON';
  }

  return { shape, approx, minX, minY, w, h, smoothed };
}

function douglasPeucker(pts, eps) {
  if (pts.length <= 2) return pts;
  let maxD = 0, idx = 0;
  const end = pts.length - 1;
  for (let i = 1; i < end; i++) {
    const d = pointLineDistance(pts[i], pts[0], pts[end]);
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD > eps) {
    const l = douglasPeucker(pts.slice(0, idx + 1), eps);
    const r = douglasPeucker(pts.slice(idx), eps);
    return [...l.slice(0, -1), ...r];
  }
  return [pts[0], pts[end]];
}

function pointLineDistance(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (dx === 0 && dy === 0) return dist(p, a);
  const t = ((p.x - a.x)*dx + (p.y - a.y)*dy) / (dx*dx + dy*dy);
  return dist(p, {x: a.x + t*dx, y: a.y + t*dy});
}

function polygonArea(pts) {
  let area = 0;
  for (let i = 0, j = pts.length-1; i < pts.length; j = i++) {
    area += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y);
  }
  return Math.abs(area / 2);
}

function drawCleanShape(result) {
  const { shape, approx, minX, minY, w, h, smoothed } = result;
  dctx.strokeStyle = color;
  dctx.lineWidth   = brushSize;
  dctx.lineCap     = 'round';
  dctx.lineJoin    = 'round';

  // Erase rough stroke area first
  const margin = brushSize + 8;
  dctx.clearRect(minX - margin, minY - margin, w + margin*2, h + margin*2);

  dctx.beginPath();
  if (shape === 'LINE') {
    dctx.moveTo(smoothed[0].x, smoothed[0].y);
    dctx.lineTo(smoothed[smoothed.length-1].x, smoothed[smoothed.length-1].y);
  } else if (shape === 'CIRCLE') {
    const cx = minX + w/2, cy = minY + h/2;
    const r  = Math.max(w, h) / 2;
    dctx.arc(cx, cy, r, 0, Math.PI*2);
  } else if (shape === 'RECTANGLE' || shape === 'SQUARE') {
    dctx.rect(minX, minY, w, h);
  } else {
    // TRIANGLE / POLYGON — draw approx contour
    dctx.moveTo(approx[0].x, approx[0].y);
    for (let i = 1; i < approx.length; i++) dctx.lineTo(approx[i].x, approx[i].y);
    dctx.closePath();
  }
  dctx.stroke();
  showBanner(shape, 'shape');
}

// ── Main MediaPipe result handler ────────────────────────────
function onResults(results) {
  const W = overlayCanvas.width;
  const H = overlayCanvas.height;

  // FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
    frameCount = 0;
    lastFrameTime = now;
    fpsLabel.textContent = `${fps} FPS`;
  }

  octx.clearRect(0, 0, W, H);

  const hands = results.multiHandLandmarks || [];

  // ── TWO HANDS → ZOOM ──────────────────────────────────────
  if (hands.length === 2) {
    const p0 = isPinching(hands[0]);
    const p1 = isPinching(hands[1]);

    if (p0 && p1) {
      const t0 = lm(hands[0], 4), i0 = lm(hands[0], 8);
      const t1 = lm(hands[1], 4), i1 = lm(hands[1], 8);
      const mid0 = {x:(t0.x+i0.x)/2, y:(t0.y+i0.y)/2};
      const mid1 = {x:(t1.x+i1.x)/2, y:(t1.y+i1.y)/2};
      const curDist = dist(mid0, mid1);

      if (!inZoom) {
        inZoom = true;
        zoomInitDist  = curDist;
        zoomInitScale = zoomScale;
      }
      const raw = zoomInitScale * (curDist / zoomInitDist);
      zoomScale = Math.max(0.5, Math.min(3.0, zoomScale * 0.8 + raw * 0.2));
      applyCanvasTransform();

      // Draw visual feedback
      octx.strokeStyle = '#00ddff';
      octx.lineWidth   = 2;
      octx.beginPath();
      octx.moveTo(mid0.x, mid0.y);
      octx.lineTo(mid1.x, mid1.y);
      octx.stroke();
      [mid0, mid1].forEach(p => {
        octx.beginPath();
        octx.arc(p.x, p.y, 10, 0, Math.PI*2);
        octx.fillStyle = '#00ddff';
        octx.fill();
      });
      const midX = (mid0.x+mid1.x)/2, midY = (mid0.y+mid1.y)/2;
      octx.fillStyle = '#00ddff';
      octx.font = 'bold 18px Segoe UI';
      octx.fillText(`Zoom ${Math.round(zoomScale*100)}%`, midX - 40, midY - 16);
      setModeLabel('ZOOM', '#00ddff');
    } else {
      inZoom = false;
    }
    prevPt = null; smoothPt = null;
    return;
  }

  inZoom = false;

  // ── ONE HAND ──────────────────────────────────────────────
  if (hands.length === 1) {
    const lms = hands[0];
    const raw = detectGesture(lms);
    const mode = debounceMode(raw);

    const indexTip = lm(lms, 8);
    const thumbTip = lm(lms, 4);

    // Smooth cursor
    if (!smoothPt) smoothPt = {...indexTip};
    else {
      smoothPt.x += (indexTip.x - smoothPt.x) * (1 - SMOOTH_ALPHA);
      smoothPt.y += (indexTip.y - smoothPt.y) * (1 - SMOOTH_ALPHA);
    }
    const ix = smoothPt.x, iy = smoothPt.y;

    // Draw hand skeleton on overlay
    drawSkeleton(octx, lms, W, H);

    // ── ERASE ───────────────────────────────────────────────
    if (mode === 'ERASE') {
      const palmPts = [0,4,8,12,16,20].map(i => lm(lms, i));
      const cx = palmPts.reduce((s,p)=>s+p.x,0)/palmPts.length;
      const cy = palmPts.reduce((s,p)=>s+p.y,0)/palmPts.length;
      snapshot();
      dctx.clearRect(cx - ERASER_R, cy - ERASER_R, ERASER_R*2, ERASER_R*2);
      octx.beginPath();
      octx.arc(cx, cy, ERASER_R, 0, Math.PI*2);
      octx.strokeStyle = '#ff4444';
      octx.lineWidth = 2;
      octx.stroke();
      setModeLabel('ERASE', '#ff5555');
      prevPt = null;
      finishStroke();

    // ── MOVE ────────────────────────────────────────────────
    } else if (mode === 'MOVE') {
      const pinching = isPinching(lms);
      const action   = updatePinchDetector(pinching);
      const midX = (ix + thumbTip.x) / 2;
      const midY = (iy + thumbTip.y) / 2;

      if (isMoveActive) {
        if (!moveAnchor) {
          moveAnchor     = {x: midX, y: midY};
          moveOffsetBase = {...canvasOffset};
        } else {
          canvasOffset.x = moveOffsetBase.x + (midX - moveAnchor.x);
          canvasOffset.y = moveOffsetBase.y + (midY - moveAnchor.y);
          applyCanvasTransform();
        }
        setModeLabel('MOVE', '#ffaa44');
      } else {
        setModeLabel('MOVE', '#ffaa44');
      }

      // Visual
      octx.beginPath();
      octx.arc(midX, midY, 14, 0, Math.PI*2);
      octx.fillStyle = 'rgba(255,160,50,0.8)';
      octx.fill();
      octx.beginPath();
      octx.moveTo(thumbTip.x, thumbTip.y);
      octx.lineTo(ix, iy);
      octx.strokeStyle = '#ffaa44';
      octx.lineWidth = 2;
      octx.stroke();
      prevPt = null;
      finishStroke();

    // ── DRAW ────────────────────────────────────────────────
    } else if (mode === 'DRAW') {
      updatePinchDetector(false);
      setModeLabel('DRAW', '#64ff96');

      // Cursor dot
      octx.beginPath();
      octx.arc(ix, iy, brushSize/2 + 3, 0, Math.PI*2);
      octx.fillStyle = color;
      octx.fill();
      octx.strokeStyle = '#fff';
      octx.lineWidth = 1;
      octx.stroke();

      if (!prevPt) {
        prevPt = {x: ix, y: iy};
        strokePoints = [{x: ix, y: iy}];
        snapshot();
      } else {
        const d = dist({x:ix,y:iy}, prevPt);
        if (d >= MIN_DRAW_PX) {
          dctx.beginPath();
          dctx.moveTo(prevPt.x, prevPt.y);
          dctx.lineTo(ix, iy);
          dctx.strokeStyle = color;
          dctx.lineWidth   = brushSize;
          dctx.lineCap     = 'round';
          dctx.lineJoin    = 'round';
          dctx.stroke();
          prevPt = {x: ix, y: iy};
          strokePoints.push({x: ix, y: iy});
        }
      }
      wasDrawing = true;

    // ── CURSOR ──────────────────────────────────────────────
    } else if (mode === 'CURSOR') {
      updatePinchDetector(false);
      setModeLabel('CURSOR', '#ffdd44');
      octx.beginPath();
      octx.arc(ix, iy, 12, 0, Math.PI*2);
      octx.strokeStyle = '#ffdd44';
      octx.lineWidth = 2;
      octx.stroke();
      octx.beginPath();
      octx.arc(ix, iy, 3, 0, Math.PI*2);
      octx.fillStyle = '#ffdd44';
      octx.fill();
      prevPt = null;
      finishStroke();

    // ── IDLE ────────────────────────────────────────────────
    } else {
      updatePinchDetector(false);
      setModeLabel('IDLE', '#888');
      prevPt = null;
      finishStroke();
    }

  } else {
    // No hands
    prevPt = null; smoothPt = null;
    updatePinchDetector(false);
    finishStroke();
    setModeLabel('IDLE', '#888');
  }
}

// Called when drawing stops — trigger shape detection
function finishStroke() {
  if (wasDrawing && strokePoints.length >= 20) {
    const result = detectShape(strokePoints);
    if (result) drawCleanShape(result);
  }
  if (wasDrawing) wasDrawing = false;
  strokePoints = [];
}

// ── Draw hand skeleton on overlay ────────────────────────────
function drawSkeleton(ctx, lms, W, H) {
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
  ];
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth   = 1.5;
  connections.forEach(([a,b]) => {
    const pa = lm(lms, a), pb = lm(lms, b);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });
  for (let i = 0; i < 21; i++) {
    const p = lm(lms, i);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI*2);
    ctx.fillStyle = i === 8 ? '#64ff96' : 'rgba(255,255,255,0.6)';
    ctx.fill();
  }
}

function setModeLabel(text, col) {
  modeLabel.textContent = text;
  modeLabel.style.color = col;
}

// ── MediaPipe Hands setup ─────────────────────────────────────
const hands = new Hands({
  locateFile: file =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands:          2,
  modelComplexity:      1,
  minDetectionConfidence: 0.75,
  minTrackingConfidence:  0.65
});

hands.onResults(onResults);

// ── Camera setup ─────────────────────────────────────────────
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    // Sync canvas pixel size once video is playing
    video.addEventListener('playing', syncCanvasSize);
    syncCanvasSize();

    // MediaPipe Camera utility — feeds frames to hands model
    const camera = new Camera(video, {
      onFrame: async () => { await hands.send({ image: video }); },
      width: 1280, height: 720
    });
    camera.start();

    statusMsg.textContent = 'Camera ready — raise your index finger to draw!';
    setTimeout(() => statusMsg.classList.add('hidden'), 3000);

  } catch (err) {
    statusMsg.textContent = `Camera error: ${err.message}`;
    statusMsg.style.color = '#ff5555';
    console.error('Camera error:', err);
  }
}

startCamera();

