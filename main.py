"""
Real-Time Hand Gesture Drawing Application  — with Depth-Based 3D Drawing
==========================================================================
Uses OpenCV + MediaPipe + MiDaS (PyTorch) for AR-style 3D air drawing.
New: YOLOv8 object detection + Blueprint Mode + Canvas Pan gesture.

Gestures (single hand):
  - Index finger only     → DRAW MODE
  - Index + Middle        → CURSOR MODE
  - Full Palm stationary  → ERASER MODE
  - Full Palm moving      → PAN MODE (slides canvas horizontally)
  - Pinch hold (thumb+index) → MOVE MODE (drag canvas)
  - Index+Middle up, tap down → UNDO
  - Index+Middle up, double-tap → REDO
  - Two hands CLAP        → CLEAR CANVAS

Gestures (two hands):
  - BOTH hands pinching   → ZOOM MODE

Controls:
  - 'c' → Clear canvas + 3D strokes
  - 's' → Save drawing as PNG
  - 'o' → Toggle object detection (YOLOv8)
  - 'b' → Toggle blueprint mode (requires object detection ON)
  - 'd' → Toggle DEPTH / 3D mode on/off
  - 'q' → Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import collections

# ── PyTorch + MiDaS (imported lazily so app still runs if torch missing) ──
try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found — depth mode disabled. "
          "Install with: pip install torch torchvision")

# ── YOLOv8 (imported lazily — app runs without it) ────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found — object detection disabled. "
          "Install with: pip install ultralytics")

# ─────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

PINCH_THRESHOLD  = 50    # px — distance to trigger pinch (MOVE or ZOOM)
ERASER_RADIUS    = 45    # px — eraser circle radius on screen
SMOOTHING_FACTOR = 0.5   # cursor EMA smoothing (higher = more lag but smoother)
MIN_DRAW_DIST    = 5     # px — minimum movement in canvas-space before drawing

ZOOM_MIN    = 0.5        # minimum zoom scale
ZOOM_MAX    = 3.0        # maximum zoom scale
ZOOM_SMOOTH = 0.80       # zoom EMA smoothing

# Gesture debounce: how many consecutive frames a gesture must hold before activating
GESTURE_DEBOUNCE_FRAMES = 3

# ── Pinch tap / hold timing ───────────────────────────────────────────────
PINCH_HOLD_SECS    = 0.5   # hold longer than this → MOVE (drag)
PINCH_TAP_MAX_SECS = 0.3   # release faster than this → counts as a tap
PINCH_DOUBLE_SECS  = 0.4   # two taps within this window → REDO
PINCH_COOLDOWN     = 0.5   # minimum gap between undo/redo actions
ACTION_LABEL_SECS  = 1.2   # how long to show UNDO / REDO banner on screen

# ── Undo / Redo history ───────────────────────────────────────────────────
UNDO_MAX_STEPS     = 30    # maximum undo levels kept in memory

# Shape detection
SHAPE_MIN_POINTS   = 20    # ignore strokes with fewer collected points
SHAPE_MIN_AREA     = 800   # ignore shapes whose bounding area is too small (px²)
SHAPE_LABEL_SECS   = 2.5   # seconds to display the detected shape name on screen
SHAPE_POLY_EPSILON = 0.03  # approxPolyDP epsilon as fraction of perimeter

# ── Object Detection + Blueprint Mode ────────────────────────────────────
YOLO_MODEL_NAME    = "yolov8n.pt"   # nano = fastest; swap for yolov8s.pt etc.
YOLO_CONF          = 0.25           # minimum detection confidence (lowered for better recall)
YOLO_INPUT_SIZE    = 640            # native YOLO resolution — better accuracy than 416
BLUEPRINT_GRID_GAP = 40             # pixels between blueprint grid lines
BLUEPRINT_BG       = (30, 10, 0)    # dark navy background (BGR)
BLUEPRINT_EDGE_CLR = (255, 220, 80) # cyan-white edge colour (BGR)

# ── Canvas Pan ────────────────────────────────────────────────────────────
PAN_THRESHOLD      = 15    # px — minimum wrist movement to trigger pan
PAN_SMOOTH         = 0.4   # EMA smoothing for pan dx (0=instant, 1=frozen)

# ── 3D / Depth drawing ────────────────────────────────────────────────────
# Approximate camera intrinsics for a typical 1280×720 webcam.
# These are reasonable defaults; a calibrated camera will give better results.
FOCAL_LENGTH_X   = 1000.0   # fx  (pixels)
FOCAL_LENGTH_Y   = 1000.0   # fy  (pixels)
# cx, cy are set dynamically from frame size in main()

DEPTH_SMOOTH_WIN = 5        # moving-average window for Z values (frames)
DEPTH_SPIKE_MAX  = 0.5      # reject depth jumps larger than this fraction
DEPTH_SCALE      = 5.0      # world-unit scale applied to MiDaS relative depth
DEPTH_MIN_Z      = 0.1      # clamp minimum Z to avoid division by zero

# ORB camera tracker
ORB_MAX_FEATURES = 300      # max ORB keypoints per frame
ORB_MATCH_RATIO  = 0.75     # Lowe ratio test threshold

# Perspective visual effect
PERSP_THICKNESS_SCALE = 2.0  # base thickness multiplier for near strokes
PERSP_COLOR_DARKEN    = 0.4  # how much to darken far strokes (0=none, 1=black)

# Color palette (BGR)
COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   255, 0),
    "Blue":   (255, 0,   0),
    "Yellow": (0,   255, 255),
    "White":  (255, 255, 255),
    "Cyan":   (255, 255, 0),
    "Purple": (255, 0,   255),
}
COLOR_NAMES = list(COLORS.keys())

# ── Floating side panel UI ───────────────────────────────────────────────
# Large panel on the LEFT side of the camera feed.
# All buttons work with mouse click AND hand gesture hover/dwell.

PANEL_X          = 10     # panel left edge
PANEL_Y          = 10     # panel top edge
PANEL_W          = 110    # panel width
PANEL_ALPHA      = 0.82   # panel background transparency

# Color buttons (circular, stacked vertically)
COLOR_BTN_R      = 22     # radius of each color circle
COLOR_BTN_GAP    = 8      # gap between circles
COLOR_BTN_X      = PANEL_X + PANEL_W // 2   # centre x of color column

# Brush size buttons
BRUSH_SIZES      = [3, 6, 10, 16]
BRUSH_BTN_W      = 90
BRUSH_BTN_H      = 32
BRUSH_BTN_X      = PANEL_X + 10
BRUSH_BTN_GAP    = 6

# Action buttons (Undo, Redo, Clear, Save)
ACTION_BTN_W     = 90
ACTION_BTN_H     = 32
ACTION_BTN_X     = PANEL_X + 10
ACTION_BTN_GAP   = 6

# Trash bin button
TRASH_BTN_W      = 90
TRASH_BTN_H      = 36
TRASH_BTN_X      = PANEL_X + 10
TRASH_DWELL_SECS = 0.8

# Legacy aliases (keep old code working)
COLOR_BTN_W      = BRUSH_BTN_W
COLOR_BTN_H      = BRUSH_BTN_H
COLOR_BTN_Y      = PANEL_Y
COLOR_BTN_MARGIN = COLOR_BTN_GAP
BRUSH_BTN_X_START = BRUSH_BTN_X
BRUSH_BTN_Y      = PANEL_Y
TRASH_BTN_Y      = PANEL_Y



# ─────────────────────────────────────────────
# MediaPipe Setup  (max_num_hands=2 for zoom)
# ─────────────────────────────────────────────
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
)

# ─────────────────────────────────────────────
# Canvas State
# base_canvas  — the permanent drawing surface (never shifted/scaled in place)
# offset_x/y   — translation applied at render time
# scale_factor — zoom applied at render time
# ─────────────────────────────────────────────
base_canvas  = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
offset_x     = 0
offset_y     = 0
scale_factor = 1.0          # current zoom level (smoothed)

prev_x, prev_y     = None, None
smooth_x, smooth_y = None, None

# MOVE state
in_pinch             = False
pinch_anchor_x       = 0
pinch_anchor_y       = 0
pinch_anchor_off_x   = 0
pinch_anchor_off_y   = 0

# ZOOM state
in_zoom              = False
zoom_initial_dist    = None
zoom_initial_scale   = 1.0
zoom_target_scale    = 1.0   # raw target before smoothing

current_color_idx = 0
current_brush_idx = 1


# ─────────────────────────────────────────────
# Helper: landmark → pixel coords
# ─────────────────────────────────────────────
def lm_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


# ─────────────────────────────────────────────
# detect_fingers()
# Returns [thumb, index, middle, ring, pinky] booleans
# ─────────────────────────────────────────────
def detect_fingers(landmarks, w, h):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb: x-axis (mirrored frame — tip.x < pip.x means extended)
    fingers.append(landmarks[tips[0]].x < landmarks[pips[0]].x)

    # Index → Pinky: tip.y < pip.y means extended
    for i in range(1, 5):
        fingers.append(landmarks[tips[i]].y < landmarks[pips[i]].y)

    return fingers  # [thumb, index, middle, ring, pinky]


# ─────────────────────────────────────────────
# is_pinching()
# True when thumb tip and index tip are closer
# than PINCH_THRESHOLD pixels.
# ─────────────────────────────────────────────
def is_pinching(landmarks, w, h):
    """Return (bool, dist) — whether this hand is in a pinch gesture."""
    thumb_tip = lm_px(landmarks, 4, w, h)
    index_tip = lm_px(landmarks, 8, w, h)
    dist = float(np.hypot(index_tip[0] - thumb_tip[0],
                          index_tip[1] - thumb_tip[1]))
    return dist < PINCH_THRESHOLD, dist


# ─────────────────────────────────────────────
# get_gesture_mode()
# Priority (strict): ERASE > MOVE > DRAW > CURSOR > IDLE
# Zoom is handled separately in main loop (two-hand path).
# ─────────────────────────────────────────────
def get_gesture_mode(fingers, landmarks, w, h):
    """
    Classify single-hand gesture.
    Returns one of: 'ERASE', 'MOVE', 'DRAW', 'CURSOR', 'IDLE'
    """
    thumb, index, middle, ring, pinky = fingers

    # 1. Full palm (all 5 fingers extended) → ERASE
    if thumb and index and middle and ring and pinky:
        return "ERASE"

    # 2. Pinch (thumb+index close, middle/ring/pinky folded) → MOVE
    pinching, _ = is_pinching(landmarks, w, h)
    if pinching and not middle and not ring and not pinky:
        return "MOVE"

    # 3. Only index extended → DRAW
    if index and not thumb and not middle and not ring and not pinky:
        return "DRAW"

    # 4. Index + middle extended → CURSOR
    if index and middle and not ring and not pinky:
        return "CURSOR"

    return "IDLE"


# ─────────────────────────────────────────────
# GestureDebouncer
# Requires a gesture to be stable for N frames
# before it becomes the active mode.
# ─────────────────────────────────────────────
class GestureDebouncer:
    def __init__(self, required_frames=GESTURE_DEBOUNCE_FRAMES):
        self.required  = required_frames
        self.candidate = "IDLE"   # gesture seen this streak
        self.streak    = 0        # consecutive frames of candidate
        self.active    = "IDLE"   # confirmed active mode

    def update(self, raw_mode):
        """
        Feed the raw detected mode each frame.
        Returns the debounced (stable) mode.
        """
        if raw_mode == self.candidate:
            self.streak += 1
        else:
            # New candidate — reset streak
            self.candidate = raw_mode
            self.streak    = 1

        # Promote candidate to active once streak is long enough
        if self.streak >= self.required:
            self.active = self.candidate

        return self.active

    def reset(self):
        self.candidate = "IDLE"
        self.streak    = 0
        self.active    = "IDLE"


# ─────────────────────────────────────────────
# ShapeDetector
# Collects canvas-space drawing points during a
# DRAW stroke, then on stroke-end classifies the
# rough shape and replaces it with a clean one.
# ─────────────────────────────────────────────
class ShapeDetector:
    def __init__(self):
        self.points      = []          # canvas-space (x, y) collected this stroke
        self.was_drawing = False       # True while DRAW mode was active last frame
        self.label       = ""          # last detected shape name
        self.label_timer = 0.0         # time.time() when label was set

    # ── called every frame ───────────────────
    def update(self, mode, cx, cy, cw, ch):
        """
        Call once per frame from the main loop.

        Parameters
        ----------
        mode         : current debounced mode string
        cx, cy       : canvas-space finger position (only used when mode==DRAW)
        cw, ch       : canvas width / height (for bounds clamping)

        Returns
        -------
        (label, new_points) : shape name and list of new points (or None)
        """
        currently_drawing = (mode == "DRAW") and (cy >= 60)

        if currently_drawing:
            # Accumulate points (cx/cy already clamped by caller)
            if len(self.points) == 0:
                self.points.append((cx, cy))
            else:
                last = self.points[-1]
                if np.hypot(cx - last[0], cy - last[1]) >= MIN_DRAW_DIST:
                    self.points.append((cx, cy))
            self.was_drawing = True

        elif self.was_drawing:
            # Mode just left DRAW — trigger detection
            self.was_drawing = False
            detected, new_points = self._detect_and_replace(cw, ch)
            self.points = []
            if detected:
                self.label       = detected
                self.label_timer = time.time()
                print(f"[SHAPE] Detected: {detected}")
                return detected, new_points

        else:
            # Not drawing and wasn't drawing — nothing to do
            self.was_drawing = False

        # Clear label after display timeout
        if self.label and (time.time() - self.label_timer > SHAPE_LABEL_SECS):
            self.label = ""

        return self.label, None

    # ── internal: classify + replace ─────────
    def _detect_and_replace(self, cw, ch):
        """
        Analyse self.points, classify the shape, and return geometric points.
        Returns (shape_name, new_points) or (None, None) if detection failed.
        """
        if len(self.points) < SHAPE_MIN_POINTS:
            return None, None

        # ── 1. Smooth points with a simple moving average (window=5) ──
        pts = np.array(self.points, dtype=np.float32)
        kernel = 5
        smoothed = []
        for i in range(len(pts)):
            lo = max(0, i - kernel // 2)
            hi = min(len(pts), i + kernel // 2 + 1)
            smoothed.append(pts[lo:hi].mean(axis=0))
        pts_smooth = np.array(smoothed, dtype=np.int32)

        # ── 2. Build contour array ─────────────────────────────────────
        contour = pts_smooth.reshape((-1, 1, 2))

        # ── 3. Bounding rect — reject tiny shapes ─────────────────────
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < SHAPE_MIN_AREA:
            return None, None

        # ── 4. Approximate polygon ────────────────────────────────────
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, SHAPE_POLY_EPSILON * peri, True)
        n      = len(approx)

        # ── 5. Classify ───────────────────────────────────────────────
        if n == 2:
            shape_name = "LINE"
            new_points = [tuple(pts_smooth[0]), tuple(pts_smooth[-1])]
        elif n == 3:
            shape_name = "TRIANGLE"
            new_points = [tuple(p[0]) for p in approx]
            new_points.append(new_points[0])
        elif n == 4:
            # Distinguish square vs rectangle by aspect ratio
            aspect = w / float(h) if h > 0 else 1.0
            shape_name = "SQUARE" if 0.85 <= aspect <= 1.15 else "RECTANGLE"
            new_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        elif n > 6:
            # Extra circularity check: area / (pi * r^2)
            area      = cv2.contourArea(contour)
            (cx2, cy2), radius = cv2.minEnclosingCircle(contour)
            if radius > 0:
                circularity = area / (np.pi * radius ** 2)
            else:
                circularity = 0.0
            if circularity > 0.65:
                shape_name = "CIRCLE"
                new_points = [tuple(p) for p in cv2.ellipse2Poly((int(cx2), int(cy2)), (int(radius), int(radius)), 0, 0, 360, 10)]
            else:
                shape_name = "POLYGON"
                new_points = [tuple(p[0]) for p in approx]
                new_points.append(new_points[0])
        else:
            shape_name = "POLYGON"
            new_points = [tuple(p[0]) for p in approx]
            new_points.append(new_points[0])

        return shape_name, new_points

    def reset(self):
        """Hard reset — call on canvas clear."""
        self.points      = []
        self.was_drawing = False
        self.label       = ""
        self.label_timer = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ██  3D DEPTH DRAWING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# DepthEstimator
# Wraps MiDaS model loading and inference.
# Falls back gracefully if PyTorch unavailable.
# ─────────────────────────────────────────────
class DepthEstimator:
    """
    Loads MiDaS 'MiDaS_small' (fastest variant) and produces
    a normalised depth map [0..1] for each BGR frame.
    """
    def __init__(self):
        self.model     = None
        self.transform = None
        self.device    = None
        self._loaded   = False

    def load(self):
        """Download / load the MiDaS small model. Call once at startup."""
        if not TORCH_AVAILABLE:
            print("[DEPTH] PyTorch unavailable — depth mode disabled.")
            return False
        try:
            print("[DEPTH] Loading MiDaS_small … (first run downloads ~80 MB)")
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")
            # torch.hub caches the model after first download
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False)
            self.model.to(self.device)
            self.model.eval()

            # Official MiDaS small transform
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False)
            self.transform = midas_transforms.small_transform
            self._loaded   = True
            print(f"[DEPTH] MiDaS_small loaded on {self.device}.")
            return True
        except Exception as exc:
            print(f"[DEPTH] Failed to load MiDaS: {exc}")
            return False

    @property
    def ready(self):
        return self._loaded

    def get_depth_map(self, bgr_frame):
        """
        Run MiDaS inference on a BGR frame.

        Returns
        -------
        depth_norm : np.ndarray float32, shape (H, W), values in [0, 1]
                     where 1 = closest, 0 = farthest.
                     Returns None if model not loaded.
        """
        if not self._loaded:
            return None

        # MiDaS expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=bgr_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]  (MiDaS output is inverse depth — larger = closer)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth   # shape (H, W)


# ─────────────────────────────────────────────
# convert_to_3D()
# Back-project a screen pixel + depth value
# into a 3D camera-space point.
# ─────────────────────────────────────────────
def convert_to_3D(px, py, depth_val, fx, fy, cx_cam, cy_cam):
    """
    Parameters
    ----------
    px, py     : pixel coordinates (screen space)
    depth_val  : normalised depth [0..1], scaled by DEPTH_SCALE
    fx, fy     : focal lengths (pixels)
    cx_cam, cy_cam : principal point (frame centre)

    Returns
    -------
    (X, Y, Z) : 3D point in camera space (float)
    """
    Z = max(depth_val * DEPTH_SCALE, DEPTH_MIN_Z)
    X = (px - cx_cam) * Z / fx
    Y = (py - cy_cam) * Z / fy
    return float(X), float(Y), float(Z)


# ─────────────────────────────────────────────
# project_3D_to_2D()
# Project a list of 3D world points back to
# screen pixels using the current camera pose.
# ─────────────────────────────────────────────
def project_3D_to_2D(points_3d, R, t, fx, fy, cx_cam, cy_cam):
    """
    Parameters
    ----------
    points_3d  : list of (X, Y, Z) world-space floats
    R          : 3×3 rotation matrix (world → camera)
    t          : 3-vector translation (world → camera)
    fx, fy, cx_cam, cy_cam : camera intrinsics

    Returns
    -------
    projected  : list of (px, py, Z_cam) — screen coords + camera-space Z
                 Points behind the camera (Z_cam <= 0) are skipped (None).
    """
    projected = []
    R_arr = np.array(R, dtype=np.float64)
    t_arr = np.array(t, dtype=np.float64).reshape(3)

    for (X, Y, Z) in points_3d:
        p_world = np.array([X, Y, Z], dtype=np.float64)
        p_cam   = R_arr @ p_world + t_arr
        Zc = p_cam[2]
        if Zc <= DEPTH_MIN_Z:
            projected.append(None)
            continue
        px = int(p_cam[0] * fx / Zc + cx_cam)
        py = int(p_cam[1] * fy / Zc + cy_cam)
        projected.append((px, py, Zc))

    return projected


# ─────────────────────────────────────────────
# CameraTracker
# Estimates camera motion between frames using
# ORB features + BFMatcher + Essential Matrix.
# Maintains a cumulative world→camera transform.
# ─────────────────────────────────────────────
class CameraTracker:
    """
    Tracks camera rotation (R) and translation (t) across frames.
    Uses ORB keypoints + BFMatcher + RANSAC Essential Matrix.
    Falls back to identity when tracking fails.
    """
    def __init__(self, fx, fy, cx_cam, cy_cam):
        self.fx     = fx
        self.fy     = fy
        self.cx     = cx_cam
        self.cy     = cy_cam

        # Intrinsic matrix
        self.K = np.array([[fx,  0, cx_cam],
                           [ 0, fy, cy_cam],
                           [ 0,  0,      1]], dtype=np.float64)

        # ORB detector + BF matcher
        self.orb     = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Cumulative world→camera pose (starts at identity)
        self.R_world = np.eye(3, dtype=np.float64)   # rotation
        self.t_world = np.zeros(3, dtype=np.float64)  # translation

        self.prev_gray = None
        self.prev_kp   = None
        self.prev_des  = None

    def update(self, bgr_frame):
        """
        Feed the current frame. Updates self.R_world and self.t_world.
        Call once per frame.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if (self.prev_gray is None or des is None or
                self.prev_des is None or len(kp) < 8):
            # Not enough data yet — keep identity
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_des  = des
            return

        # Match descriptors with Lowe ratio test
        raw_matches = self.matcher.knnMatch(self.prev_des, des, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ORB_MATCH_RATIO * n.distance:
                    good.append(m)

        if len(good) < 8:
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_des  = des
            return

        pts_prev = np.float32(
            [self.prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32(
            [kp[m.trainIdx].pt for m in good])

        # Essential matrix → relative R, t
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_des  = des
            return

        _, R_rel, t_rel, _ = cv2.recoverPose(
            E, pts_curr, pts_prev, self.K, mask=mask)

        # Accumulate: new_world = R_rel @ old_world
        # t_rel is unit-length (scale unknown from monocular) — we use a
        # small fixed step to avoid drift from scale ambiguity.
        TRANSLATION_SCALE = 0.02
        self.R_world = R_rel @ self.R_world
        self.t_world = R_rel @ self.t_world + t_rel.flatten() * TRANSLATION_SCALE

        self.prev_gray = gray
        self.prev_kp   = kp
        self.prev_des  = des

    @property
    def pose(self):
        """Return (R, t) — current world→camera transform."""
        return self.R_world.copy(), self.t_world.copy()


# ─────────────────────────────────────────────
# DepthSmoother
# Per-pixel moving-average filter on depth maps
# to reduce temporal noise and spike rejection.
# ─────────────────────────────────────────────
class DepthSmoother:
    """
    Maintains a rolling buffer of depth maps and returns
    the per-pixel mean, with spike rejection.
    """
    def __init__(self, window=DEPTH_SMOOTH_WIN):
        self.window = window
        self.buf    = collections.deque(maxlen=window)
        self.last   = None

    def update(self, depth_map):
        """
        Feed a new normalised depth map.
        Returns the smoothed depth map (same shape).
        """
        if self.last is not None:
            # Spike rejection: if change > threshold, blend instead of replace
            diff = np.abs(depth_map - self.last)
            mask = diff > DEPTH_SPIKE_MAX
            depth_map = np.where(mask,
                                 0.5 * self.last + 0.5 * depth_map,
                                 depth_map)
        self.buf.append(depth_map)
        self.last = depth_map
        return np.mean(self.buf, axis=0).astype(np.float32)

    def get_depth_at(self, x, y):
        """
        Return smoothed depth value at pixel (x, y).
        Returns 0.5 (mid-range) if no data yet.
        """
        if self.last is None:
            return 0.5
        h, w = self.last.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        return float(self.last[y, x])


# ─────────────────────────────────────────────
# Stroke3D
# Stores one continuous 3D drawing stroke as a
# list of world-space (X, Y, Z) points plus the
# color and base brush thickness.
# ─────────────────────────────────────────────
class Stroke3D:
    def __init__(self, color, brush):
        self.points = []   # list of (X, Y, Z) world-space floats
        self.color  = color
        self.brush  = brush

    def add_point(self, X, Y, Z):
        self.points.append((float(X), float(Y), float(Z)))

    def __len__(self):
        return len(self.points)


# ─────────────────────────────────────────────
# render_3D_lines()
# Project all 3D strokes to screen and draw them
# with perspective-correct thickness and colour.
# ─────────────────────────────────────────────
def render_3D_lines(frame, strokes_3d, R, t, fx, fy, cx_cam, cy_cam):
    """
    Draw all 3D strokes onto `frame` in-place.

    Perspective effects:
      - Thickness  ∝ 1/Z  (near = thick, far = thin)
      - Brightness ∝ 1/Z  (near = vivid, far = darker)

    Parameters
    ----------
    frame      : BGR frame to draw on
    strokes_3d : list of Stroke3D objects
    R, t       : current world→camera pose
    fx, fy, cx_cam, cy_cam : camera intrinsics
    """
    fh, fw = frame.shape[:2]

    # Compute a reference Z for normalising perspective (median of all points)
    all_z = []
    for stroke in strokes_3d:
        for (_, _, Z) in stroke.points:
            all_z.append(Z)
    ref_z = float(np.median(all_z)) if all_z else DEPTH_SCALE * 0.5
    ref_z = max(ref_z, DEPTH_MIN_Z)

    for stroke in strokes_3d:
        if len(stroke.points) < 2:
            continue

        projected = project_3D_to_2D(
            stroke.points, R, t, fx, fy, cx_cam, cy_cam)

        for i in range(1, len(projected)):
            p0 = projected[i - 1]
            p1 = projected[i]
            if p0 is None or p1 is None:
                continue

            px0, py0, z0 = p0
            px1, py1, z1 = p1
            z_avg = max((z0 + z1) / 2.0, DEPTH_MIN_Z)

            # Skip if both endpoints are off-screen
            if (px0 < -fw or px0 > 2 * fw or py0 < -fh or py0 > 2 * fh):
                continue

            # Perspective thickness: closer = thicker
            persp_scale = ref_z / z_avg
            thickness   = max(1, int(stroke.brush * PERSP_THICKNESS_SCALE
                                     * persp_scale))
            thickness   = min(thickness, 30)   # cap to avoid huge blobs

            # Perspective colour: closer = brighter
            darken = float(np.clip(
                1.0 - PERSP_COLOR_DARKEN * (z_avg / ref_z - 1.0), 0.2, 1.0))
            b = int(stroke.color[0] * darken)
            g = int(stroke.color[1] * darken)
            r = int(stroke.color[2] * darken)

            cv2.line(frame,
                     (int(px0), int(py0)),
                     (int(px1), int(py1)),
                     (b, g, r), thickness,
                     lineType=cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# ██  END OF 3D PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# PinchActionDetector
# Interprets pinch timing to distinguish:
#   - Short tap  (<0.3 s)  → UNDO
#   - Double tap (<0.4 s between taps) → REDO
#   - Long hold  (>0.5 s)  → MOVE (drag & drop)
#
# Returns one of: 'MOVE', 'UNDO', 'REDO', None
# ─────────────────────────────────────────────
class PinchActionDetector:
    """
    Watches thumb+index pinch transitions.
    ONLY handles MOVE (hold > 0.5s).
    UNDO/REDO are now handled by TwoFingerTapDetector (index+middle).
    """
    def __init__(self):
        self._active     = False
        self._start_time = 0.0
        self.is_move     = False
        self.action_label       = ""
        self._action_label_time = 0.0

    def update(self, pinching_now):
        """
        Returns 'MOVE' when hold threshold crossed, else None.
        """
        now    = time.time()
        action = None

        if pinching_now and not self._active:
            self._active     = True
            self._start_time = now

        elif pinching_now and self._active:
            if now - self._start_time >= PINCH_HOLD_SECS and not self.is_move:
                self.is_move = True
                action = "MOVE"
                self._set_label("MOVE MODE")

        elif not pinching_now and self._active:
            self._active = False
            if self.is_move:
                self.is_move = False
            self._start_time = 0.0

        if self.action_label and now - self._action_label_time > ACTION_LABEL_SECS:
            self.action_label = ""

        return action

    def _set_label(self, text):
        self.action_label       = text
        self._action_label_time = time.time()

    def reset(self):
        self._active     = False
        self._start_time = 0.0
        self.is_move     = False
        self.action_label = ""


# ─────────────────────────────────────────────
# TwoFingerTapDetector
# Detects index + middle finger tap gestures
# for UNDO (single tap) and REDO (double tap).
#
# "Tap" = index AND middle fingers quickly
# dip down (tips drop below a threshold) and
# come back up within TAP_MAX_SECS.
# ─────────────────────────────────────────────
class TwoFingerTapDetector:
    """
    Monitors the vertical position of index (8) and middle (12) fingertips.
    When both tips dip below their MCP joints simultaneously and recover
    quickly, it counts as a tap.

    Single tap  → UNDO
    Double tap  → REDO  (second tap within PINCH_DOUBLE_SECS)
    """
    def __init__(self):
        self._down       = False   # fingers currently dipped
        self._down_start = 0.0
        self._tap_count  = 0
        self._last_tap   = 0.0
        self._last_action = 0.0
        self.action_label       = ""
        self._action_label_time = 0.0

    def _fingers_dipped(self, landmarks):
        """
        True when BOTH index and middle tips are BELOW their MCP joints.
        This is the opposite of the normal 'up' state — a deliberate dip.
        """
        idx_dipped = landmarks[8].y  > landmarks[5].y  + 0.04
        mid_dipped = landmarks[12].y > landmarks[9].y  + 0.04
        return idx_dipped and mid_dipped

    def update(self, landmarks):
        """
        Call every frame when in CURSOR mode (index+middle up).

        Parameters
        ----------
        landmarks : MediaPipe hand landmark list

        Returns
        -------
        action : 'UNDO', 'REDO', or None
        """
        now    = time.time()
        action = None
        dipped = self._fingers_dipped(landmarks)

        if dipped and not self._down:
            # Fingers just dipped
            self._down       = True
            self._down_start = now

        elif not dipped and self._down:
            # Fingers just came back up
            dur = now - self._down_start
            self._down = False
            if dur < PINCH_TAP_MAX_SECS:
                # Quick dip = tap
                if (self._tap_count == 1 and
                        now - self._last_tap < PINCH_DOUBLE_SECS):
                    # Double tap → REDO
                    if now - self._last_action >= PINCH_COOLDOWN:
                        action = "REDO"
                        self._set_label("REDO")
                        self._last_action = now
                    self._tap_count = 0
                else:
                    self._tap_count = 1
                    self._last_tap  = now

        # Single-tap window expired → UNDO
        if (not dipped and self._tap_count == 1 and
                now - self._last_tap >= PINCH_DOUBLE_SECS):
            if now - self._last_action >= PINCH_COOLDOWN:
                action = "UNDO"
                self._set_label("UNDO")
                self._last_action = now
            self._tap_count = 0

        if self.action_label and now - self._action_label_time > ACTION_LABEL_SECS:
            self.action_label = ""

        return action

    def _set_label(self, text):
        self.action_label       = text
        self._action_label_time = time.time()

    def reset(self):
        self._down       = False
        self._down_start = 0.0
        self._tap_count  = 0
        self._last_tap   = 0.0
        self._last_action = 0.0
        self.action_label = ""


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# ClapDetector
# Detects a clap using the classic fast-approach
# pattern: hands far apart → suddenly very close.
#
# Logic (per spec):
#   prev_distance > CLAP_HIGH  AND
#   curr_distance < CLAP_LOW   AND
#   transition happens within CLAP_FRAMES frames
# → clap_detected = True → CLEAR canvas
# ─────────────────────────────────────────────
CLAP_HIGH     = 150   # px — hands must have been this far apart before clap
CLAP_LOW      = 60    # px — hands must come this close to trigger clap
CLAP_FRAMES   = 8     # max frames the approach must happen within
CLAP_COOLDOWN = 1.5   # seconds between clap triggers


def detect_two_hands(lms_list):
    """Return True when exactly two hands are detected."""
    return len(lms_list) == 2


def calculate_hand_distance(lms0, lms1, fw, fh):
    """
    Compute horizontal distance between wrist (landmark 0) of two hands.
    Returns absolute pixel distance.
    """
    w0 = lm_px(lms0, 0, fw, fh)
    w1 = lm_px(lms1, 0, fw, fh)
    return abs(w0[0] - w1[0])


def detect_clap(prev_distance, current_distance):
    """
    Return True when hands snapped together quickly.
    prev_distance > CLAP_HIGH and current_distance < CLAP_LOW.
    """
    return prev_distance > CLAP_HIGH and current_distance < CLAP_LOW


def clear_canvas(canvas):
    """Wipe the entire drawing canvas to black."""
    canvas[:] = 0
    return canvas


class ClapDetector:
    """
    Stateful clap detector.

    Tracks the last N frames of hand distance.
    Fires 'CLEAR' when the fast-approach pattern is detected:
      - Any of the last CLAP_FRAMES distances was > CLAP_HIGH
      - Current distance < CLAP_LOW
      - Cooldown has elapsed
    """
    def __init__(self):
        self._dist_history  = collections.deque(maxlen=CLAP_FRAMES)
        self._last_clap     = 0.0
        self._was_two_hands = False   # track 2→1 merge as fallback
        self.action_label       = ""
        self._action_label_time = 0.0

    def update(self, lms_list, fw, fh):
        """
        Call every frame.

        Parameters
        ----------
        lms_list : list of landmark lists (pass [] for no hands, [lms] for 1)
        fw, fh   : frame width / height

        Returns
        -------
        'CLEAR' if clap detected, else None
        """
        now    = time.time()
        action = None

        # Clear label after timeout
        if self.action_label and now - self._action_label_time > ACTION_LABEL_SECS:
            self.action_label = ""

        n = len(lms_list)

        # ── Two hands visible ─────────────────────────────────
        if n == 2:
            d = calculate_hand_distance(lms_list[0], lms_list[1], fw, fh)
            self._dist_history.append(d)
            self._was_two_hands = True

            # Check clap: current distance very small AND
            # at least one recent distance was large
            if d < CLAP_LOW and len(self._dist_history) >= 2:
                max_recent = max(self._dist_history)
                if detect_clap(max_recent, d):
                    if now - self._last_clap >= CLAP_COOLDOWN:
                        action = "CLEAR"
                        self._last_clap = now
                        self._set_label("CLAP! CANVAS CLEARED")
                        self._dist_history.clear()

        # ── Went from 2 hands to 1 (merge = clap moment) ─────
        elif n == 1 and self._was_two_hands:
            self._was_two_hands = False
            # If last known distances show a fast approach, fire clap
            if len(self._dist_history) >= 2:
                max_recent = max(self._dist_history)
                last_d     = self._dist_history[-1]
                if detect_clap(max_recent, last_d):
                    if now - self._last_clap >= CLAP_COOLDOWN:
                        action = "CLEAR"
                        self._last_clap = now
                        self._set_label("CLAP! CANVAS CLEARED")
            self._dist_history.clear()

        # ── No hands ─────────────────────────────────────────
        else:
            self._was_two_hands = False
            self._dist_history.clear()

        return action

    def _set_label(self, text):
        self.action_label       = text
        self._action_label_time = time.time()

    def reset(self):
        self._dist_history.clear()
        self._last_clap     = 0.0
        self._was_two_hands = False
        self.action_label   = ""


# ─────────────────────────────────────────────
# UndoRedoManager
# Maintains a history of canvas snapshots and
# 3D stroke lists for undo / redo operations.
# ─────────────────────────────────────────────
class UndoRedoManager:
    """
    Stores canvas + 3D stroke state snapshots.
    Call snapshot() after each completed stroke.
    Call undo() / redo() to step through history.
    """
    def __init__(self, max_steps=UNDO_MAX_STEPS):
        self.max_steps  = max_steps
        self._undo_stack = []   # list of (canvas_copy, strokes_3d_copy)
        self._redo_stack = []   # list of (canvas_copy, strokes_3d_copy)

    def snapshot(self, canvas, strokes_3d):
        """Save current state. Clears redo stack."""
        self._undo_stack.append(
            (canvas.copy(), list(strokes_3d)))
        if len(self._undo_stack) > self.max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, canvas, strokes_3d):
        """
        Restore previous state.
        Returns (new_canvas, new_strokes_3d) or current values if nothing to undo.
        """
        if not self._undo_stack:
            print("[UNDO] Nothing to undo.")
            return canvas, strokes_3d
        # Push current state onto redo stack
        self._redo_stack.append((canvas.copy(), list(strokes_3d)))
        prev_canvas, prev_strokes = self._undo_stack.pop()
        print(f"[UNDO] Restored. ({len(self._undo_stack)} steps remain)")
        return prev_canvas.copy(), list(prev_strokes)

    def redo(self, canvas, strokes_3d):
        """
        Re-apply an undone state.
        Returns (new_canvas, new_strokes_3d) or current values if nothing to redo.
        """
        if not self._redo_stack:
            print("[REDO] Nothing to redo.")
            return canvas, strokes_3d
        # Push current state onto undo stack
        self._undo_stack.append((canvas.copy(), list(strokes_3d)))
        next_canvas, next_strokes = self._redo_stack.pop()
        print(f"[REDO] Restored. ({len(self._redo_stack)} redo steps remain)")
        return next_canvas.copy(), list(next_strokes)

    def reset(self):
        self._undo_stack.clear()
        self._redo_stack.clear()


# ═══════════════════════════════════════════════════════════════════════════
# ██  OBJECT-LEVEL STROKE MANAGEMENT  (Features 2, 3, 4)
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# Stroke2D
# One complete 2D drawing stroke stored as a
# list of canvas-space (x, y) points.
# Supports per-object drag and scale.
# ─────────────────────────────────────────────
class Stroke2D:
    """A single drawn stroke with its own color, thickness, and point list."""
    def __init__(self, color, thickness):
        self.points    = []          # list of (x, y) canvas-space ints
        self.color     = color       # BGR tuple
        self.thickness = thickness   # px

    def add_point(self, x, y):
        self.points.append((int(x), int(y)))

    def center(self):
        """Return (cx, cy) centroid of all points."""
        if not self.points:
            return (0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def bounding_box(self):
        """Return (x1, y1, x2, y2) tight bounding box."""
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def distance_to(self, px, py):
        """Minimum distance from point (px, py) to any line segment in this stroke."""
        if not self.points:
            return float('inf')
        if len(self.points) == 1:
            return float(np.hypot(self.points[0][0] - px, self.points[0][1] - py))
        
        min_d = float('inf')
        p = np.array([px, py], dtype=np.float64)
        for i in range(1, len(self.points)):
            a = np.array(self.points[i - 1], dtype=np.float64)
            b = np.array(self.points[i], dtype=np.float64)
            ab = b - a
            ap = p - a
            if ab[0] == 0 and ab[1] == 0:
                d = np.linalg.norm(ap)
            else:
                t = np.dot(ap, ab) / np.dot(ab, ab)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
        return float(min_d)

    def draw(self, canvas):
        """Render this stroke onto canvas."""
        if len(self.points) < 2:
            if self.points:
                cv2.circle(canvas, self.points[0],
                           max(self.thickness // 2, 1),
                           self.color, -1, lineType=cv2.LINE_AA)
            return
        for i in range(1, len(self.points)):
            cv2.line(canvas, self.points[i - 1], self.points[i],
                     self.color, self.thickness, lineType=cv2.LINE_AA)

    def draw_selected(self, canvas):
        """Render with a selection highlight (dashed bounding box)."""
        self.draw(canvas)
        x1, y1, x2, y2 = self.bounding_box()
        pad = 8
        cv2.rectangle(canvas,
                      (x1 - pad, y1 - pad),
                      (x2 + pad, y2 + pad),
                      (0, 255, 200), 1, lineType=cv2.LINE_AA)

    def __len__(self):
        return len(self.points)


# ─────────────────────────────────────────────
# Standalone functions (per spec)
# ─────────────────────────────────────────────

def start_new_stroke(color, thickness):
    """Create and return a new empty Stroke2D."""
    return Stroke2D(color, thickness)


def update_stroke(stroke, x, y):
    """
    Add point (x, y) to stroke if it moved enough.
    Returns True if a point was added.
    """
    if not stroke.points:
        stroke.add_point(x, y)
        return True
    last = stroke.points[-1]
    if np.hypot(x - last[0], y - last[1]) >= MIN_DRAW_DIST:
        stroke.add_point(x, y)
        return True
    return False


def end_stroke(stroke):
    """
    Finalise a stroke. Returns the stroke if it has points, else None.
    Caller should append to the stroke list.
    """
    return stroke if stroke and len(stroke) > 0 else None


def select_object(strokes, px, py, max_dist=60):
    """
    Find the stroke closest to canvas point (px, py).

    Parameters
    ----------
    strokes  : list of Stroke2D
    px, py   : canvas-space query point
    max_dist : ignore strokes farther than this (px)

    Returns
    -------
    index of selected stroke, or -1 if none within max_dist
    """
    best_idx  = -1
    best_dist = max_dist
    for i, s in enumerate(strokes):
        d = s.distance_to(px, py)
        if d < best_dist:
            best_dist = d
            best_idx  = i
    return best_idx


def move_object(stroke, dx, dy):
    """
    Translate all points of stroke by (dx, dy).
    Modifies in-place.
    """
    stroke.points = [(p[0] + int(dx), p[1] + int(dy))
                     for p in stroke.points]


def scale_object(stroke, scale_factor):
    """
    Scale stroke points relative to their centroid.

    Parameters
    ----------
    stroke       : Stroke2D to scale
    scale_factor : float multiplier (>1 = zoom in, <1 = zoom out)
    """
    if not stroke.points:
        return
    cx, cy = stroke.center()
    stroke.points = [
        (int(cx + (p[0] - cx) * scale_factor),
         int(cy + (p[1] - cy) * scale_factor))
        for p in stroke.points
    ]


# ─────────────────────────────────────────────
# StrokeManager
# Owns the list of all 2D strokes.
# Handles: active stroke building, selection,
# drag, per-object zoom, undo/redo snapshots,
# and rendering to a canvas.
# ─────────────────────────────────────────────
class StrokeManager:
    """
    Central manager for all 2D drawing strokes.

    Replaces direct drawing onto base_canvas for 2D strokes.
    The canvas is rebuilt from strokes each frame (or on change).
    """
    def __init__(self):
        self.strokes        = []      # list of completed Stroke2D
        self.active_stroke  = None    # stroke being drawn right now
        self.selected_idx   = -1      # index of selected stroke (-1 = none)

        # Object drag state
        self._drag_anchor_x = 0       # canvas-space x when drag started
        self._drag_anchor_y = 0
        self._drag_pts_snap = []      # snapshot of selected stroke points at drag start

        # Object zoom state
        self.obj_zoom_active   = False
        self._obj_zoom_init_d  = 0.0
        self._obj_zoom_pts_snap = []  # snapshot of selected stroke points at zoom start

    # ── Drawing ──────────────────────────────────────────────
    def begin_draw(self, color, thickness):
        """Start a new stroke. Call when DRAW mode begins."""
        self.active_stroke = start_new_stroke(color, thickness)
        self.selected_idx  = -1   # deselect on new draw

    def add_draw_point(self, x, y):
        """Add a point to the active stroke. Returns True if point added."""
        if self.active_stroke is None:
            return False
        return update_stroke(self.active_stroke, x, y)

    def finish_draw(self):
        """
        Finalise the active stroke and add to list.
        Returns the completed Stroke2D or None.
        """
        s = end_stroke(self.active_stroke)
        self.active_stroke = None
        if s:
            self.strokes.append(s)
        return s

    # ── Selection ────────────────────────────────────────────
    def try_select(self, px, py):
        """
        Try to select a stroke near (px, py).
        Returns selected index or -1.
        """
        self.selected_idx = select_object(self.strokes, px, py)
        return self.selected_idx

    # ── Drag ─────────────────────────────────────────────────
    def begin_drag(self, px, py):
        """Record drag anchor when pinch starts on a selected stroke."""
        self._drag_anchor_x = px
        self._drag_anchor_y = py
        if self.selected_idx >= 0:
            self._drag_pts_snap = list(self.strokes[self.selected_idx].points)

    def update_drag(self, px, py):
        """Move selected stroke to follow finger. Call every frame during drag."""
        if self.selected_idx < 0:
            return
        dx = px - self._drag_anchor_x
        dy = py - self._drag_anchor_y
        # Restore snapshot + apply total delta (avoids drift)
        self.strokes[self.selected_idx].points = [
            (p[0] + int(dx), p[1] + int(dy))
            for p in self._drag_pts_snap
        ]

    def end_drag(self):
        """Release drag."""
        self._drag_pts_snap = []

    # ── Object zoom ──────────────────────────────────────────
    def begin_obj_zoom(self, init_dist):
        """Record initial pinch distance for object zoom."""
        if self.selected_idx < 0:
            return
        self.obj_zoom_active    = True
        self._obj_zoom_init_d   = max(init_dist, 1.0)
        self._obj_zoom_pts_snap = list(self.strokes[self.selected_idx].points)

    def update_obj_zoom(self, curr_dist):
        """Scale selected stroke by ratio of current/initial pinch distance."""
        if self.selected_idx < 0 or not self.obj_zoom_active:
            return
        raw_scale = curr_dist / self._obj_zoom_init_d
        raw_scale = float(np.clip(raw_scale, 0.1, 10.0))
        s = self.strokes[self.selected_idx]
        # Compute centroid from snapshot
        if not self._obj_zoom_pts_snap:
            return
        cx = int(np.mean([p[0] for p in self._obj_zoom_pts_snap]))
        cy = int(np.mean([p[1] for p in self._obj_zoom_pts_snap]))
        s.points = [
            (int(cx + (p[0] - cx) * raw_scale),
             int(cy + (p[1] - cy) * raw_scale))
            for p in self._obj_zoom_pts_snap
        ]

    def end_obj_zoom(self):
        self.obj_zoom_active    = False
        self._obj_zoom_pts_snap = []

    # ── Rendering ────────────────────────────────────────────
    def render(self, canvas):
        """
        Redraw all strokes onto canvas (clears first).
        Call whenever strokes change.
        """
        canvas[:] = 0
        for i, s in enumerate(self.strokes):
            if i == self.selected_idx:
                s.draw_selected(canvas)
            else:
                s.draw(canvas)
        if self.active_stroke:
            self.active_stroke.draw(canvas)

    def render_overlay(self, canvas):
        """
        Draw only the active (in-progress) stroke on top of canvas.
        More efficient than full re-render every frame.
        """
        if self.active_stroke:
            self.active_stroke.draw(canvas)

    # ── Undo / Redo support ──────────────────────────────────
    def snapshot(self):
        """Return a deep copy of current strokes for undo stack."""
        import copy
        return copy.deepcopy(self.strokes)

    def restore(self, snap):
        """Restore strokes from a snapshot."""
        import copy
        self.strokes       = copy.deepcopy(snap)
        self.selected_idx  = -1
        self.active_stroke = None

    # ── Misc ─────────────────────────────────────────────────
    def clear(self):
        self.strokes        = []
        self.active_stroke  = None
        self.selected_idx   = -1
        self.obj_zoom_active = False

    def delete_selected(self):
        """Remove the currently selected stroke."""
        if 0 <= self.selected_idx < len(self.strokes):
            self.strokes.pop(self.selected_idx)
            self.selected_idx = -1


# ─────────────────────────────────────────────
# draw_on_canvas()
# Smooth line with jitter filter
# ─────────────────────────────────────────────
def draw_on_canvas(canvas, x, y, prev_x, prev_y, color, brush_size):
    if prev_x is not None and prev_y is not None:
        if np.hypot(x - prev_x, y - prev_y) >= MIN_DRAW_DIST:
            cv2.line(canvas, (prev_x, prev_y), (x, y),
                     color, brush_size, lineType=cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x, y), max(brush_size // 2, 1),
                   color, -1, lineType=cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────
# zoom_canvas()
# Called only when BOTH hands are confirmed pinching.
# Measures distance between the two pinch midpoints
# (midpoint of thumb+index on each hand) to drive scale.
# ─────────────────────────────────────────────
def zoom_canvas(lms1, lms2, w, h,
                in_zoom, zoom_initial_dist,
                zoom_initial_scale, zoom_target_scale, scale_factor):
    """
    Compute updated scale_factor from two-hand pinch spread.

    Uses the midpoint between thumb and index on each hand as the
    zoom anchor point — more stable than raw fingertip distance.

    Returns
    -------
    (in_zoom, zoom_initial_dist, zoom_initial_scale,
     zoom_target_scale, scale_factor, pt1, pt2)
    """
    # Pinch midpoint for each hand (thumb tip + index tip midpoint)
    t1 = lm_px(lms1, 4, w, h)
    i1 = lm_px(lms1, 8, w, h)
    t2 = lm_px(lms2, 4, w, h)
    i2 = lm_px(lms2, 8, w, h)

    pt1 = ((t1[0] + i1[0]) // 2, (t1[1] + i1[1]) // 2)  # hand-1 pinch centre
    pt2 = ((t2[0] + i2[0]) // 2, (t2[1] + i2[1]) // 2)  # hand-2 pinch centre

    current_dist = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    current_dist = max(current_dist, 1.0)

    if not in_zoom:
        # First frame of zoom — record anchor
        zoom_initial_dist  = current_dist
        zoom_initial_scale = scale_factor
        in_zoom = True

    # Raw scale relative to when zoom started
    raw_scale         = zoom_initial_scale * (current_dist / zoom_initial_dist)
    zoom_target_scale = float(np.clip(raw_scale, ZOOM_MIN, ZOOM_MAX))

    # Smooth with EMA
    scale_factor = ZOOM_SMOOTH * scale_factor + (1.0 - ZOOM_SMOOTH) * zoom_target_scale
    scale_factor = float(np.clip(scale_factor, ZOOM_MIN, ZOOM_MAX))

    return (in_zoom, zoom_initial_dist, zoom_initial_scale,
            zoom_target_scale, scale_factor, pt1, pt2)


# ─────────────────────────────────────────────
# render_canvas()
# Apply offset + zoom and composite onto frame.
# Non-destructive: base_canvas is never modified here.
# ─────────────────────────────────────────────
def render_canvas(frame, base_canvas, offset_x, offset_y, scale_factor):
    """
    Scale base_canvas by scale_factor, then place it on frame
    with (offset_x, offset_y) translation.
    Black pixels are transparent.
    """
    fh, fw = frame.shape[:2]
    ch, cw = base_canvas.shape[:2]

    # Scale the canvas
    new_w = max(1, int(cw * scale_factor))
    new_h = max(1, int(ch * scale_factor))
    scaled = cv2.resize(base_canvas, (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR)

    # Centre offset: when scale=1 offset=0 → canvas fills frame exactly
    # We anchor the scaled canvas so its centre aligns with frame centre + offset
    cx = fw // 2 + offset_x
    cy = fh // 2 + offset_y
    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Clamp to frame bounds and compute corresponding source ROI
    sx1 = max(0, -x1);  sy1 = max(0, -y1)
    dx1 = max(0,  x1);  dy1 = max(0,  y1)
    dx2 = min(fw, x2);  dy2 = min(fh, y2)
    sx2 = sx1 + (dx2 - dx1)
    sy2 = sy1 + (dy2 - dy1)

    if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
        return frame   # canvas fully off-screen

    roi_canvas = scaled[sy1:sy2, sx1:sx2]
    roi_frame  = frame[dy1:dy2, dx1:dx2]

    # Mask: non-black canvas pixels overwrite frame
    gray = cv2.cvtColor(roi_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi_frame,  roi_frame,  mask=mask_inv)
    fg = cv2.bitwise_and(roi_canvas, roi_canvas, mask=mask)
    frame[dy1:dy2, dx1:dx2] = cv2.add(bg, fg)
    return frame


# ─────────────────────────────────────────────
# screen_to_canvas()
# Convert a screen pixel (sx, sy) to canvas pixel
# accounting for current offset and scale.
# ─────────────────────────────────────────────
def screen_to_canvas(sx, sy, fw, fh, cw, ch, offset_x, offset_y, scale_factor):
    """Map a screen coordinate back to the base_canvas coordinate."""
    # Canvas top-left on screen
    cx_origin = fw // 2 + offset_x - int(cw * scale_factor) // 2
    cy_origin = fh // 2 + offset_y - int(ch * scale_factor) // 2
    cx = int((sx - cx_origin) / scale_factor)
    cy = int((sy - cy_origin) / scale_factor)
    return cx, cy


# ─────────────────────────────────────────────
# draw_ui()
# ─────────────────────────────────────────────
def draw_ui(frame, current_color_idx, current_brush_idx,
            mode, fps, scale_factor, shape_label="",
            depth_enabled=False, action_label="",
            trash_hover_progress=0.0, trash_has_selection=False):
    """
    Draw the floating left-side control panel over the camera frame.
    All buttons are large and work with mouse click or hand gesture dwell.
    """
    h, w = frame.shape[:2]

    # ── Compute panel height dynamically ─────────────────────────────────
    n_colors  = len(COLOR_NAMES)
    n_brushes = len(BRUSH_SIZES)
    n_actions = 4   # Undo, Redo, Clear, Save

    # Section heights
    sec_title_h  = 22
    color_sec_h  = sec_title_h + n_colors * (COLOR_BTN_R * 2 + COLOR_BTN_GAP) + 8
    brush_sec_h  = sec_title_h + n_brushes * (BRUSH_BTN_H + BRUSH_BTN_GAP) + 8
    action_sec_h = sec_title_h + n_actions * (ACTION_BTN_H + ACTION_BTN_GAP) + 8
    trash_sec_h  = TRASH_BTN_H + 12
    panel_h      = color_sec_h + brush_sec_h + action_sec_h + trash_sec_h + 16

    # ── Semi-transparent panel background ────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (PANEL_X - 4, PANEL_Y - 4),
                  (PANEL_X + PANEL_W + 4, PANEL_Y + panel_h + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

    # Panel border
    cv2.rectangle(frame,
                  (PANEL_X - 4, PANEL_Y - 4),
                  (PANEL_X + PANEL_W + 4, PANEL_Y + panel_h + 4),
                  (80, 80, 80), 1)

    cy = PANEL_Y + 8   # current y cursor

    # ── Section: COLORS ───────────────────────────────────────────────────
    cv2.putText(frame, "COLOR", (PANEL_X + 28, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cy += sec_title_h

    for i, name in enumerate(COLOR_NAMES):
        cx_btn = COLOR_BTN_X
        cy_btn = cy + COLOR_BTN_R
        col    = COLORS[name]

        # Circle fill
        cv2.circle(frame, (cx_btn, cy_btn), COLOR_BTN_R, col, -1)

        # Selection ring
        if i == current_color_idx:
            cv2.circle(frame, (cx_btn, cy_btn), COLOR_BTN_R + 4,
                       (255, 255, 255), 2)
            cv2.circle(frame, (cx_btn, cy_btn), COLOR_BTN_R + 7,
                       (200, 200, 200), 1)
        else:
            cv2.circle(frame, (cx_btn, cy_btn), COLOR_BTN_R + 2,
                       (80, 80, 80), 1)

        # Color name label to the right
        cv2.putText(frame, name[:4],
                    (cx_btn + COLOR_BTN_R + 6, cy_btn + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200) if i != current_color_idx else (255, 255, 255),
                    1)

        cy += COLOR_BTN_R * 2 + COLOR_BTN_GAP

    cy += 8

    # ── Section: BRUSH SIZE ───────────────────────────────────────────────
    cv2.putText(frame, "SIZE", (PANEL_X + 32, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cy += sec_title_h

    size_labels = ["S", "M", "L", "XL"]
    for i, (size, lbl) in enumerate(zip(BRUSH_SIZES, size_labels)):
        bx = BRUSH_BTN_X
        by = cy
        is_sel = (i == current_brush_idx)
        bg_col = (60, 60, 60) if not is_sel else (80, 120, 80)
        cv2.rectangle(frame, (bx, by),
                      (bx + BRUSH_BTN_W, by + BRUSH_BTN_H), bg_col, -1)
        if is_sel:
            cv2.rectangle(frame, (bx - 1, by - 1),
                          (bx + BRUSH_BTN_W + 1, by + BRUSH_BTN_H + 1),
                          (100, 220, 100), 2)
        else:
            cv2.rectangle(frame, (bx, by),
                          (bx + BRUSH_BTN_W, by + BRUSH_BTN_H),
                          (70, 70, 70), 1)

        # Dot preview
        dot_x = bx + 18
        dot_y = by + BRUSH_BTN_H // 2
        cv2.circle(frame, (dot_x, dot_y), min(size, 10), (200, 200, 200), -1)

        # Label
        cv2.putText(frame, lbl,
                    (bx + 34, by + BRUSH_BTN_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if is_sel else (180, 180, 180), 1)
        cv2.putText(frame, f"{size}px",
                    (bx + 55, by + BRUSH_BTN_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (160, 160, 160), 1)

        cy += BRUSH_BTN_H + BRUSH_BTN_GAP

    cy += 8

    # ── Section: ACTIONS ──────────────────────────────────────────────────
    cv2.putText(frame, "ACTIONS", (PANEL_X + 18, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cy += sec_title_h

    actions = [
        ("UNDO",  (60, 60, 100), (120, 120, 220)),
        ("REDO",  (60, 100, 60), (120, 220, 120)),
        ("CLEAR", (80, 60, 60),  (200, 100, 100)),
        ("SAVE",  (60, 80, 60),  (100, 200, 100)),
    ]
    for (lbl, bg, fg) in actions:
        bx = ACTION_BTN_X
        by = cy
        cv2.rectangle(frame, (bx, by),
                      (bx + ACTION_BTN_W, by + ACTION_BTN_H), bg, -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + ACTION_BTN_W, by + ACTION_BTN_H), fg, 1)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, lbl,
                    (bx + (ACTION_BTN_W - tw) // 2, by + ACTION_BTN_H // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg, 1, cv2.LINE_AA)
        cy += ACTION_BTN_H + ACTION_BTN_GAP

    cy += 4

    # ── Trash bin ─────────────────────────────────────────────────────────
    bx = TRASH_BTN_X
    by = cy
    if trash_has_selection:
        t_bg = (30, 30, 160); t_fg = (80, 80, 255); t_bdr = (0, 0, 255)
    else:
        t_bg = (50, 50, 50);  t_fg = (160, 160, 160); t_bdr = (100, 100, 100)

    cv2.rectangle(frame, (bx, by),
                  (bx + TRASH_BTN_W, by + TRASH_BTN_H), t_bg, -1)
    cv2.rectangle(frame, (bx, by),
                  (bx + TRASH_BTN_W, by + TRASH_BTN_H), t_bdr, 2)

    # Trash icon
    ix0 = bx + 12; iy0 = by + 10
    cv2.rectangle(frame, (ix0, iy0 + 6), (ix0 + 18, iy0 + 20), t_fg, 2)
    cv2.rectangle(frame, (ix0 - 2, iy0 + 2), (ix0 + 20, iy0 + 6), t_fg, 2)
    cv2.rectangle(frame, (ix0 + 5, iy0 - 2), (ix0 + 13, iy0 + 2), t_fg, 2)
    for lx in [ix0 + 5, ix0 + 9, ix0 + 13]:
        cv2.line(frame, (lx, iy0 + 8), (lx, iy0 + 18), t_fg, 1)

    # Label
    lbl_del = "DEL SELECTED" if trash_has_selection else "DEL LAST"
    cv2.putText(frame, lbl_del,
                (bx + 34, by + TRASH_BTN_H // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, t_fg, 1, cv2.LINE_AA)

    # Dwell arc
    if trash_hover_progress > 0:
        angle = int(360 * trash_hover_progress)
        cx_t  = bx + TRASH_BTN_W // 2
        cy_t  = by + TRASH_BTN_H // 2
        cv2.ellipse(frame, (cx_t, cy_t),
                    (TRASH_BTN_W // 2 + 5, TRASH_BTN_H // 2 + 5),
                    -90, 0, angle,
                    (0, 0, 255) if trash_has_selection else (0, 200, 255), 3)

    cy += TRASH_BTN_H + 8

    # ── Top-right: Mode + FPS ─────────────────────────────────────────────
    mode_colors = {
        "DRAW":   (0, 255, 100),  "CURSOR": (255, 200, 0),
        "ERASE":  (0, 100, 255),  "MOVE":   (255, 100, 0),
        "ZOOM":   (0, 220, 255),  "IDLE":   (150, 150, 150),
    }
    mode_lbl = mode
    if mode == "ERASE":   mode_lbl = "ERASE (PALM)"
    elif mode == "ZOOM":  mode_lbl = f"ZOOM {int(scale_factor*100)}%"
    mc = mode_colors.get(mode, (200, 200, 200))

    # Mode pill top-right
    (mw, mh), _ = cv2.getTextSize(mode_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    mx = w - mw - 20
    my = 14
    cv2.rectangle(frame, (mx - 8, my - mh - 4), (mx + mw + 8, my + 6),
                  (20, 20, 20), -1)
    cv2.rectangle(frame, (mx - 8, my - mh - 4), (mx + mw + 8, my + 6),
                  mc, 1)
    cv2.putText(frame, mode_lbl, (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mc, 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 70, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # 3D depth indicator
    if depth_enabled:
        cv2.putText(frame, "3D ON", (w - 70, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

    # ── Action banner (UNDO/REDO/MOVE) ────────────────────────────────────
    if action_label:
        label_colors = {
            "UNDO": (80, 80, 255), "REDO": (80, 200, 80),
            "MOVE MODE": (255, 140, 0),
        }
        ac = label_colors.get(action_label, (220, 220, 220))
        (tw, th), _ = cv2.getTextSize(action_label,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        lx = (w - tw) // 2; ly = h // 2
        cv2.rectangle(frame, (lx-20, ly-th-16), (lx+tw+20, ly+16),
                      (15, 15, 15), -1)
        cv2.rectangle(frame, (lx-20, ly-th-16), (lx+tw+20, ly+16), ac, 3)
        cv2.putText(frame, action_label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, ac, 4, cv2.LINE_AA)

    # ── Shape label ───────────────────────────────────────────────────────
    if shape_label:
        txt = f"Shape: {shape_label}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        lx = (w - tw) // 2; ly = 120
        cv2.rectangle(frame, (lx-14, ly-th-10), (lx+tw+14, ly+10),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (lx-14, ly-th-10), (lx+tw+14, ly+10),
                      (0, 220, 120), 2)
        cv2.putText(frame, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 120), 3,
                    cv2.LINE_AA)

    return frame

def _panel_button_rects(frame_h=720):
    """
    Compute bounding rects for every panel button.
    Returns dict: key -> (x1, y1, x2, y2)
    """
    rects = {}
    cy = PANEL_Y + 8 + 22   # skip title

    # Color buttons
    for i in range(len(COLOR_NAMES)):
        cx_btn = COLOR_BTN_X
        cy_btn = cy + COLOR_BTN_R
        r = COLOR_BTN_R + 4
        rects[f'color_{i}'] = (cx_btn - r, cy_btn - r, cx_btn + r, cy_btn + r)
        cy += COLOR_BTN_R * 2 + COLOR_BTN_GAP
    cy += 8 + 22   # gap + brush title

    # Brush buttons
    for i in range(len(BRUSH_SIZES)):
        rects[f'brush_{i}'] = (BRUSH_BTN_X, cy,
                                BRUSH_BTN_X + BRUSH_BTN_W, cy + BRUSH_BTN_H)
        cy += BRUSH_BTN_H + BRUSH_BTN_GAP
    cy += 8 + 22   # gap + action title

    # Action buttons
    for lbl in ['UNDO', 'REDO', 'CLEAR', 'SAVE']:
        rects[f'action_{lbl}'] = (ACTION_BTN_X, cy,
                                   ACTION_BTN_X + ACTION_BTN_W, cy + ACTION_BTN_H)
        cy += ACTION_BTN_H + ACTION_BTN_GAP
    cy += 4

    # Trash
    rects['trash'] = (TRASH_BTN_X, cy,
                      TRASH_BTN_X + TRASH_BTN_W, cy + TRASH_BTN_H)
    return rects


def check_ui_click(ix, iy, current_color_idx, current_brush_idx):
    """Check if (ix, iy) hits a color or brush button. Returns updated indices."""
    rects = _panel_button_rects()
    for i in range(len(COLOR_NAMES)):
        x1, y1, x2, y2 = rects[f'color_{i}']
        if x1 <= ix <= x2 and y1 <= iy <= y2:
            return i, current_brush_idx
    for i in range(len(BRUSH_SIZES)):
        x1, y1, x2, y2 = rects[f'brush_{i}']
        if x1 <= ix <= x2 and y1 <= iy <= y2:
            return current_color_idx, i
    return current_color_idx, current_brush_idx


def check_action_click(ix, iy):
    """
    Check if (ix, iy) hits an action button.
    Returns action string ('UNDO','REDO','CLEAR','SAVE') or None.
    """
    rects = _panel_button_rects()
    for lbl in ['UNDO', 'REDO', 'CLEAR', 'SAVE']:
        x1, y1, x2, y2 = rects[f'action_{lbl}']
        if x1 <= ix <= x2 and y1 <= iy <= y2:
            return lbl
    return None


def is_over_trash(ix, iy, frame_w=0):
    """Return True when (ix, iy) is over the trash bin button."""
    rects = _panel_button_rects()
    x1, y1, x2, y2 = rects['trash']
    return x1 <= ix <= x2 and y1 <= iy <= y2


def is_over_panel(ix, iy):
    """Return True when (ix, iy) is anywhere inside the panel."""
    rects = _panel_button_rects()
    all_x2 = max(r[2] for r in rects.values())
    all_y2 = max(r[3] for r in rects.values())
    return (PANEL_X - 4 <= ix <= PANEL_X + PANEL_W + 4 and
            PANEL_Y - 4 <= iy <= all_y2 + 4)


# ═══════════════════════════════════════════════════════════════════════════
# ██  DRAG-TO-DELETE TRASH BIN
# ═══════════════════════════════════════════════════════════════════════════

# Trash bin geometry (top-centre of frame)
DRAG_TRASH_W    = 80    # hit-box width  (px)
DRAG_TRASH_H    = 80    # hit-box height (px)
DRAG_TRASH_Y    = 20    # distance from top of frame

def _drag_trash_rect(frame_w):
    """Return (x1, y1, x2, y2) of the drag-to-delete trash bin."""
    cx = frame_w // 2
    x1 = cx - DRAG_TRASH_W // 2
    y1 = DRAG_TRASH_Y
    return x1, y1, x1 + DRAG_TRASH_W, y1 + DRAG_TRASH_H


def draw_trash_icon(frame, is_hovering):
    """
    Draw the drag-to-delete trash bin at the top-centre of `frame`.

    Parameters
    ----------
    frame       : BGR frame (drawn on in-place)
    is_hovering : bool — True when the dragged stroke is over the bin

    Visual:
      Normal  : grey icon, normal size
      Hovering: red icon, 1.3× scale, pulsing glow ring
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _drag_trash_rect(fw)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if is_hovering:
        # Scale 1.3× around centre
        scale  = 1.3
        hw     = int(DRAG_TRASH_W * scale / 2)
        hh     = int(DRAG_TRASH_H * scale / 2)
        rx1, ry1 = cx - hw, cy - hh
        rx2, ry2 = cx + hw, cy + hh
        bg_col   = (0, 0, 180)      # dark red bg
        icon_col = (60, 60, 255)    # bright red icon
        # Glow ring
        cv2.circle(frame, (cx, cy), hw + 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), hw + 14, (0, 0, 180), 1, cv2.LINE_AA)
    else:
        rx1, ry1 = x1, y1
        rx2, ry2 = x2, y2
        bg_col   = (40, 40, 40)
        icon_col = (160, 160, 160)

    # Background pill
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), bg_col, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2),
                  (0, 0, 255) if is_hovering else (90, 90, 90), 2, cv2.LINE_AA)

    # ── Trash can icon ────────────────────────────────────────────────────
    # Scale icon elements relative to box size
    bw = rx2 - rx1
    bh = ry2 - ry1
    # Can body
    bx0 = rx1 + bw // 5
    by0 = ry1 + bh // 3
    bw2 = bw * 3 // 5
    bh2 = bh * 5 // 9
    cv2.rectangle(frame, (bx0, by0), (bx0 + bw2, by0 + bh2), icon_col, 2)
    # Lid
    cv2.rectangle(frame,
                  (bx0 - 3, by0 - 6),
                  (bx0 + bw2 + 3, by0 - 1), icon_col, 2)
    # Handle
    hx = rx1 + bw // 2
    cv2.rectangle(frame, (hx - 6, by0 - 11), (hx + 6, by0 - 6), icon_col, 2)
    # Stripes
    for lx in [bx0 + bw2 // 4, bx0 + bw2 // 2, bx0 + 3 * bw2 // 4]:
        cv2.line(frame, (lx, by0 + 4), (lx, by0 + bh2 - 4), icon_col, 1)

    # Label
    lbl = "DROP TO DELETE" if is_hovering else "DRAG HERE"
    (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, lbl,
                (cx - tw // 2, ry2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 80, 255) if is_hovering else (140, 140, 140),
                1, cv2.LINE_AA)


def is_hovering_trash(fx, fy, frame_w):
    """
    Return True when finger/pinch point (fx, fy) is inside the trash bin.

    Parameters
    ----------
    fx, fy   : screen-space finger coordinates
    frame_w  : frame width (used to compute bin centre)
    """
    x1, y1, x2, y2 = _drag_trash_rect(frame_w)
    return x1 <= fx <= x2 and y1 <= fy <= y2


def delete_stroke(stroke_mgr_obj, selected_id):
    """
    Remove the stroke with `selected_id` from stroke_mgr_obj.strokes.
    Deselects and returns True if deleted, False if not found.

    Parameters
    ----------
    stroke_mgr_obj : StrokeManager instance
    selected_id    : id attribute of the Stroke2D to remove
                     (uses index if id not available)
    """
    before = len(stroke_mgr_obj.strokes)
    stroke_mgr_obj.strokes = [
        s for s in stroke_mgr_obj.strokes
        if id(s) != selected_id
    ]
    stroke_mgr_obj.selected_idx = -1
    return len(stroke_mgr_obj.strokes) < before

# ═══════════════════════════════════════════════════════════════════════════
# ██  FEATURE 1 — OBJECT DETECTION + BLUEPRINT MODE
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# ObjectDetector
# Wraps YOLOv8 nano for real-time detection.
# Loads lazily on first call to enable().
# ─────────────────────────────────────────────
class ObjectDetector:
    """
    Thin wrapper around YOLOv8 for real-time object detection.
    Falls back gracefully if ultralytics is not installed.
    """
    def __init__(self):
        self._model   = None
        self._ready   = False
        self._enabled = False

    def enable(self):
        """Load the YOLO model (downloads ~6 MB on first run)."""
        if not YOLO_AVAILABLE:
            print("[YOLO] ultralytics not installed — object detection unavailable.")
            return False
        if self._ready:
            self._enabled = True
            return True
        try:
            print(f"[YOLO] Loading {YOLO_MODEL_NAME} …")
            self._model  = YOLO(YOLO_MODEL_NAME)
            self._ready  = True
            self._enabled = True
            print("[YOLO] Model ready.")
            return True
        except Exception as exc:
            print(f"[YOLO] Failed to load model: {exc}")
            return False

    def toggle(self):
        if not self._ready:
            return self.enable()
        self._enabled = not self._enabled
        print(f"[YOLO] Object detection {'ON' if self._enabled else 'OFF'}.")
        return self._enabled

    @property
    def enabled(self):
        return self._enabled and self._ready

    def detect_objects(self, frame):
        """
        Run YOLOv8 inference on `frame`.

        Returns
        -------
        list of dicts: [{'label': str, 'conf': float,
                          'x1': int, 'y1': int, 'x2': int, 'y2': int}, ...]
        Empty list if model not ready or no detections.
        """
        if not self.enabled:
            return []

        fh, fw = frame.shape[:2]

        # Pass frame directly — YOLO handles resizing internally
        # and preserves aspect ratio (letterboxing), giving better accuracy
        results = self._model(
            frame,
            conf=YOLO_CONF,
            imgsz=YOLO_INPUT_SIZE,
            verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'label': r.names[int(box.cls[0])],
                    'conf' : float(box.conf[0]),
                    'x1'   : int(np.clip(x1, 0, fw)),
                    'y1'   : int(np.clip(y1, 0, fh)),
                    'x2'   : int(np.clip(x2, 0, fw)),
                    'y2'   : int(np.clip(y2, 0, fh)),
                })
        return detections


def extract_roi(frame, det):
    """
    Extract the Region of Interest from `frame` using detection bounding box.

    Parameters
    ----------
    frame : BGR frame
    det   : detection dict with x1, y1, x2, y2

    Returns
    -------
    roi   : cropped BGR image, or None if bbox is degenerate
    """
    fh, fw = frame.shape[:2]
    x1 = max(0, det['x1'])
    y1 = max(0, det['y1'])
    x2 = min(fw, det['x2'])
    y2 = min(fh, det['y2'])
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def apply_blueprint_effect(roi):
    """
    Convert an ROI into an engineering blueprint style image.

    Pipeline:
      1. Dark navy background
      2. Grayscale → Gaussian blur → Canny edges
      3. Dilate edges slightly for visibility
      4. Draw engineering grid lines
      5. Overlay cyan-white edges on dark background

    Parameters
    ----------
    roi : BGR image (the object region)

    Returns
    -------
    blueprint : BGR image, same size as roi
    """
    h, w = roi.shape[:2]

    # ── 1. Dark background ────────────────────────────────────
    blueprint = np.full((h, w, 3), BLUEPRINT_BG, dtype=np.uint8)

    # ── 2. Edge detection ─────────────────────────────────────
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 40, 120)

    # Dilate edges slightly so they're visible at small sizes
    kernel = np.ones((2, 2), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # ── 3. Colour the edges ───────────────────────────────────
    edge_layer = np.zeros((h, w, 3), dtype=np.uint8)
    edge_layer[edges > 0] = BLUEPRINT_EDGE_CLR

    # ── 4. Engineering grid lines ─────────────────────────────
    grid_color = (60, 30, 10)   # subtle dark-blue grid (BGR)
    for gx in range(0, w, BLUEPRINT_GRID_GAP):
        cv2.line(blueprint, (gx, 0), (gx, h), grid_color, 1)
    for gy in range(0, h, BLUEPRINT_GRID_GAP):
        cv2.line(blueprint, (0, gy), (w, gy), grid_color, 1)

    # ── 5. Composite edges over background ────────────────────
    mask     = edges > 0
    blueprint[mask] = edge_layer[mask]

    return blueprint


def overlay_blueprint(frame, blueprint, det):
    """
    Replace the detection bounding-box region in `frame` with `blueprint`.
    Draws a cyan bounding box and label on top.

    Parameters
    ----------
    frame     : BGR frame (modified in-place)
    blueprint : BGR blueprint image (same size as ROI)
    det       : detection dict

    Returns
    -------
    frame     : modified frame
    """
    fh, fw = frame.shape[:2]
    x1 = max(0, det['x1'])
    y1 = max(0, det['y1'])
    x2 = min(fw, det['x2'])
    y2 = min(fh, det['y2'])
    bh, bw = blueprint.shape[:2]

    # Resize blueprint to match actual (clamped) bbox size
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < 2 or roi_h < 2:
        return frame
    if bw != roi_w or bh != roi_h:
        blueprint = cv2.resize(blueprint, (roi_w, roi_h))

    frame[y1:y2, x1:x2] = blueprint

    # Bounding box + label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    label_txt = f"{det['label']} {det['conf']:.0%}"
    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (255, 200, 0), -1)
    cv2.putText(frame, label_txt, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_detections(frame, detections):
    """
    Draw plain bounding boxes + labels when blueprint mode is OFF.
    """
    for det in detections:
        cv2.rectangle(frame,
                      (det['x1'], det['y1']),
                      (det['x2'], det['y2']),
                      (0, 200, 255), 2)
        label_txt = f"{det['label']} {det['conf']:.0%}"
        cv2.putText(frame, label_txt,
                    (det['x1'] + 3, det['y1'] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# ██  FEATURE 2 — CANVAS PAN (OPEN PALM SLIDE)
# ═══════════════════════════════════════════════════════════════════════════

def detect_open_palm(fingers):
    """
    Return True when all five fingers are extended (open palm).
    `fingers` is the [thumb, index, middle, ring, pinky] bool list
    from detect_fingers().
    """
    return all(fingers)


def calculate_dx(prev_x, curr_x, smooth_dx):
    """
    Compute smoothed horizontal delta between two wrist positions.

    Parameters
    ----------
    prev_x    : previous wrist x (pixels)
    curr_x    : current  wrist x (pixels)
    smooth_dx : previous smoothed dx value

    Returns
    -------
    (raw_dx, new_smooth_dx)
    """
    raw_dx    = curr_x - prev_x
    new_smooth = smooth_dx * PAN_SMOOTH + raw_dx * (1.0 - PAN_SMOOTH)
    return raw_dx, new_smooth


def shift_canvas(canvas, dx):
    """
    Shift the entire drawing canvas horizontally by `dx` pixels.
    Uses affine warp so empty areas are filled with black.

    Parameters
    ----------
    canvas : base_canvas ndarray (modified in-place)
    dx     : integer pixel shift (positive = right, negative = left)

    Returns
    -------
    shifted canvas (same shape)
    """
    h, w = canvas.shape[:2]
    M       = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(canvas, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    return shifted


# ─────────────────────────────────────────────
# Main Application Loop
# ─────────────────────────────────────────────
def main():
    global base_canvas, offset_x, offset_y, scale_factor
    global prev_x, prev_y, smooth_x, smooth_y
    global in_pinch, pinch_anchor_x, pinch_anchor_y
    global pinch_anchor_off_x, pinch_anchor_off_y
    global in_zoom, zoom_initial_dist, zoom_initial_scale
    global zoom_target_scale
    global current_color_idx, current_brush_idx

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # force DirectShow on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # ── Mouse state (for click-based panel interaction) ───────────────────
    mouse_state = {'x': -1, 'y': -1, 'clicked': False}

    def on_mouse(event, x, y, flags, param):
        mouse_state['x'] = x
        mouse_state['y'] = y
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state['clicked'] = True

    cv2.namedWindow("Hand Gesture Drawing  [d=3D  c=clear  s=save  q=quit]")
    cv2.setMouseCallback(
        "Hand Gesture Drawing  [d=3D  c=clear  s=save  q=quit]", on_mouse)

    fh_init = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw_init = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cx_cam  = fw_init / 2.0
    cy_cam  = fh_init / 2.0

    # ── 3D pipeline objects ───────────────────────────────────────────────
    depth_estimator = DepthEstimator()
    depth_smoother  = DepthSmoother(window=DEPTH_SMOOTH_WIN)
    cam_tracker     = CameraTracker(FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                                    cx_cam, cy_cam)

    strokes_3d     = []       # list of Stroke3D — world-space strokes
    current_stroke = None     # Stroke3D being built this draw session
    depth_enabled  = False    # toggled with 'd' key
    depth_ready    = False    # True once MiDaS loaded successfully

    # Z-value moving average for the fingertip (reduces jitter)
    z_history = collections.deque(maxlen=DEPTH_SMOOTH_WIN)

    # ── Gesture / shape objects ───────────────────────────────────────────
    single_debouncer = GestureDebouncer(GESTURE_DEBOUNCE_FRAMES)
    zoom_debouncer   = GestureDebouncer(GESTURE_DEBOUNCE_FRAMES)
    shape_detector   = ShapeDetector()
    pinch_detector      = PinchActionDetector()    # hold → MOVE only
    two_finger_detector = TwoFingerTapDetector()   # tap/double-tap → UNDO/REDO
    clap_detector       = ClapDetector()           # clap → CLEAR
    undo_manager        = UndoRedoManager()        # canvas + 3D stroke history

    # ── Stroke manager (object-level 2D drawing) ──────────────────────────
    stroke_mgr        = StrokeManager()
    stroke_undo_stack = []   # list of stroke snapshots for undo
    stroke_redo_stack = []   # list of stroke snapshots for redo
    STROKE_UNDO_MAX   = 30

    # Object drag state (used in MOVE mode)
    obj_drag_active   = False   # True while dragging a selected stroke
    obj_drag_started  = False   # first frame of drag

    # Object zoom state (used in TWO HANDS mode when one stroke selected)
    obj_zoom_init_dist = 0.0

    # ── Feature 1: Object detection + Blueprint ───────────────────────────
    obj_detector   = ObjectDetector()
    blueprint_mode = False   # 'b' toggles blueprint rendering
    detections     = []      # last YOLO results (list of dicts)

    # ── Feature 2: Canvas pan ─────────────────────────────────────────────
    pan_prev_wrist_x = None   # wrist x from previous frame (screen px)
    pan_smooth_dx    = 0.0    # EMA-smoothed horizontal delta

    # ── Trash bin dwell state ─────────────────────────────────────────────
    trash_dwell_start = 0.0   # time.time() when finger entered trash zone
    trash_hovering    = False  # True while finger is over trash bin

    # ── Drag-to-delete state ──────────────────────────────────────────────
    drag_trash_visible  = False   # show top-centre trash only while dragging
    drag_trash_hovering = False   # finger is over the drag-trash bin
    drag_trash_obj_id   = -1      # Python id() of the stroke being dragged

    # ── Stroke undo/redo helpers (closures over stroke_mgr stacks) ────────
    def stroke_snap():
        stroke_undo_stack.append(stroke_mgr.snapshot())
        if len(stroke_undo_stack) > STROKE_UNDO_MAX:
            stroke_undo_stack.pop(0)
        stroke_redo_stack.clear()

    def stroke_undo():
        if not stroke_undo_stack:
            return
        stroke_redo_stack.append(stroke_mgr.snapshot())
        stroke_mgr.restore(stroke_undo_stack.pop())
        stroke_mgr.render(base_canvas)

    def stroke_redo():
        if not stroke_redo_stack:
            return
        stroke_undo_stack.append(stroke_mgr.snapshot())
        stroke_mgr.restore(stroke_redo_stack.pop())
        stroke_mgr.render(base_canvas)

    prev_time = time.time()
    mode = "IDLE"

    print("=" * 65)
    print("  Hand Gesture Drawing App  (3D Depth Edition)")
    print("  Single hand : DRAW / ERASE / MOVE / CURSOR")
    print("  Both hands pinching : ZOOM IN / OUT")
    print("  'o' = toggle object detection (YOLOv8)")
    print("  'b' = toggle blueprint mode")
    print("  'd' = toggle 3D depth mode (loads MiDaS on first press)")
    print("  'c' = clear   's' = save   'q' = quit")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        ch, cw = base_canvas.shape[:2]
        cx_cam = fw / 2.0
        cy_cam = fh / 2.0

        # ── Mouse click on panel ──────────────────────────────────────────
        if mouse_state['clicked']:
            mx, my = mouse_state['x'], mouse_state['y']
            mouse_state['clicked'] = False
            # Color / brush selection
            current_color_idx, current_brush_idx = check_ui_click(
                mx, my, current_color_idx, current_brush_idx)
            # Action buttons
            act = check_action_click(mx, my)
            if act == 'UNDO':
                stroke_undo()
                base_canvas, strokes_3d = undo_manager.undo(base_canvas, strokes_3d)
            elif act == 'REDO':
                stroke_redo()
                base_canvas, strokes_3d = undo_manager.redo(base_canvas, strokes_3d)
            elif act == 'CLEAR':
                clear_canvas(base_canvas)
                stroke_mgr.clear()
                stroke_undo_stack.clear()
                stroke_redo_stack.clear()
                strokes_3d = []; current_stroke = None
                z_history.clear()
                offset_x, offset_y = 0, 0
                scale_factor = 1.0; zoom_target_scale = 1.0
                single_debouncer.reset(); zoom_debouncer.reset()
                pinch_detector.reset(); two_finger_detector.reset()
                undo_manager.reset(); shape_detector.reset()
                pan_prev_wrist_x = None; pan_smooth_dx = 0.0
                obj_drag_active = False; obj_drag_started = False
                print("[INFO] Canvas cleared by mouse click.")
            elif act == 'SAVE':
                fname = f"drawing_{int(time.time())}.png"
                cv2.imwrite(fname, base_canvas)
                print(f"[INFO] Saved '{fname}'")
            # Trash bin
            if is_over_trash(mx, my):
                if stroke_mgr.selected_idx >= 0:
                    stroke_snap(); stroke_mgr.delete_selected()
                    stroke_mgr.render(base_canvas)
                    print("[INFO] Selected stroke deleted by mouse click.")
                elif stroke_mgr.strokes:
                    stroke_snap(); stroke_mgr.strokes.pop()
                    stroke_mgr.render(base_canvas)
                    print("[INFO] Last stroke deleted by mouse click.")

        # ── 3D: update camera tracker every frame (cheap ORB) ────────────
        cam_tracker.fx = FOCAL_LENGTH_X
        cam_tracker.fy = FOCAL_LENGTH_Y
        cam_tracker.cx = cx_cam
        cam_tracker.cy = cy_cam
        cam_tracker.K  = np.array([[FOCAL_LENGTH_X, 0, cx_cam],
                                   [0, FOCAL_LENGTH_Y, cy_cam],
                                   [0, 0, 1]], dtype=np.float64)
        cam_tracker.update(frame)
        R_cam, t_cam = cam_tracker.pose

        # ── 3D: run depth estimation if enabled ──────────────────────────
        smoothed_depth = None
        if depth_enabled and depth_ready:
            raw_depth = depth_estimator.get_depth_map(frame)
            if raw_depth is not None:
                smoothed_depth = depth_smoother.update(raw_depth)

        # ── Feature 1: Object detection (YOLOv8) ─────────────────────────
        if obj_detector.enabled:
            detections = obj_detector.detect_objects(frame)
        else:
            detections = []

        # ── MediaPipe hand detection ──────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        mode    = "IDLE"
        n_hands = (len(results.multi_hand_landmarks)
                   if results.multi_hand_landmarks else 0)

        # ══════════════════════════════════════════════════════
        # TWO HANDS — ZOOM (both must be pinching)
        # ══════════════════════════════════════════════════════
        if n_hands == 2:
            lms_list = [h.landmark for h in results.multi_hand_landmarks]

            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            pinch0, _ = is_pinching(lms_list[0], fw, fh)
            pinch1, _ = is_pinching(lms_list[1], fw, fh)
            raw_two    = "ZOOM" if (pinch0 and pinch1) else "IDLE"
            stable_two = zoom_debouncer.update(raw_two)

            # Clap detection (runs every frame with two hands)
            clap_action = clap_detector.update(lms_list, fw, fh)
            if clap_action == "CLEAR":
                clear_canvas(base_canvas)
                stroke_mgr.clear()
                stroke_undo_stack.clear()
                stroke_redo_stack.clear()
                strokes_3d     = []
                current_stroke = None
                z_history.clear()
                offset_x, offset_y = 0, 0
                scale_factor       = 1.0
                zoom_target_scale  = 1.0
                single_debouncer.reset()
                zoom_debouncer.reset()
                pinch_detector.reset()
                two_finger_detector.reset()
                undo_manager.reset()
                shape_detector.reset()
                pan_prev_wrist_x = None
                pan_smooth_dx    = 0.0
                obj_drag_active  = False
                obj_drag_started = False
                print("[INFO] Canvas cleared by CLAP gesture.")
            else:
                # Show live hand distance as clap guide (only when not zooming)
                if not (pinch0 and pinch1):
                    _d = calculate_hand_distance(lms_list[0], lms_list[1], fw, fh)
                    _col = (0, 255, 100) if _d > CLAP_HIGH else (0, 100, 255)
                    cv2.putText(frame, f"Clap dist: {int(_d)}px",
                                (10, fh - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _col, 1,
                                cv2.LINE_AA)

            if stable_two == "ZOOM":
                mode = "ZOOM"
                single_debouncer.reset()

                # ── Feature 3: Object zoom vs canvas zoom ─────────────────
                # If a stroke is selected, zoom only that object.
                # Otherwise zoom the whole canvas as before.
                t0 = lm_px(lms_list[0], 4, fw, fh)
                i0 = lm_px(lms_list[0], 8, fw, fh)
                t1 = lm_px(lms_list[1], 4, fw, fh)
                i1 = lm_px(lms_list[1], 8, fw, fh)
                m0 = ((t0[0] + i0[0]) // 2, (t0[1] + i0[1]) // 2)
                m1 = ((t1[0] + i1[0]) // 2, (t1[1] + i1[1]) // 2)
                curr_pinch_dist = float(np.hypot(m0[0] - m1[0], m0[1] - m1[1]))

                if stroke_mgr.selected_idx >= 0:
                    # Object zoom
                    if not stroke_mgr.obj_zoom_active:
                        stroke_mgr.begin_obj_zoom(curr_pinch_dist)
                    else:
                        stroke_mgr.update_obj_zoom(curr_pinch_dist)
                        stroke_mgr.render(base_canvas)
                    cv2.putText(frame, "OBJECT ZOOM",
                                (fw // 2 - 70, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 220, 255), 2, cv2.LINE_AA)
                else:
                    # Canvas zoom (existing behaviour)
                    stroke_mgr.end_obj_zoom()
                    (in_zoom, zoom_initial_dist, zoom_initial_scale,
                     zoom_target_scale, scale_factor,
                     pt1, pt2) = zoom_canvas(
                        lms_list[0], lms_list[1], fw, fh,
                        in_zoom, zoom_initial_dist, zoom_initial_scale,
                        zoom_target_scale, scale_factor)
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(frame, f"Zoom: {int(scale_factor * 100)}%",
                                (mid[0] - 45, mid[1] - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

                cv2.line(frame, m0, m1, (0, 220, 255), 2)
                cv2.circle(frame, m0, 12, (0, 220, 255), -1)
                cv2.circle(frame, m1, 12, (0, 220, 255), -1)
            else:
                in_zoom = False
                zoom_initial_dist = None
                stroke_mgr.end_obj_zoom()
                mode = "IDLE"

            prev_x, prev_y     = None, None
            smooth_x, smooth_y = None, None
            in_pinch           = False
            # Finalise any open 3D stroke
            if current_stroke is not None and len(current_stroke) > 0:
                strokes_3d.append(current_stroke)
            current_stroke = None

        # ══════════════════════════════════════════════════════
        # ONE HAND — single-hand gestures
        # ══════════════════════════════════════════════════════
        elif n_hands == 1:
            # Check for clap-merge (was 2 hands, now 1 = hands clapped together)
            clap_action = clap_detector.update(
                [h.landmark for h in results.multi_hand_landmarks], fw, fh)
            if clap_action == "CLEAR":
                clear_canvas(base_canvas)
                stroke_mgr.clear()
                stroke_undo_stack.clear()
                stroke_redo_stack.clear()
                strokes_3d     = []
                current_stroke = None
                z_history.clear()
                offset_x, offset_y = 0, 0
                scale_factor       = 1.0
                zoom_target_scale  = 1.0
                single_debouncer.reset()
                zoom_debouncer.reset()
                pinch_detector.reset()
                two_finger_detector.reset()
                undo_manager.reset()
                shape_detector.reset()
                pan_prev_wrist_x = None
                pan_smooth_dx    = 0.0
                obj_drag_active  = False
                obj_drag_started = False
                print("[INFO] Canvas cleared by CLAP gesture.")
            in_zoom = False
            zoom_initial_dist = None
            zoom_debouncer.reset()

            hand_lms  = results.multi_hand_landmarks[0]
            landmarks = hand_lms.landmark

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            fingers  = detect_fingers(landmarks, fw, fh)
            raw_mode = get_gesture_mode(fingers, landmarks, fw, fh)
            mode     = single_debouncer.update(raw_mode)

            index_tip = lm_px(landmarks, 8, fw, fh)
            thumb_tip = lm_px(landmarks, 4, fw, fh)

            if smooth_x is None:
                smooth_x, smooth_y = float(index_tip[0]), float(index_tip[1])
            else:
                smooth_x += (index_tip[0] - smooth_x) * (1.0 - SMOOTHING_FACTOR)
                smooth_y += (index_tip[1] - smooth_y) * (1.0 - SMOOTHING_FACTOR)
            ix, iy = int(smooth_x), int(smooth_y)
            smooth_thumb_x = float(thumb_tip[0])
            smooth_thumb_y = float(thumb_tip[1])

            # ── ERASE ─────────────────────────────────────────
            if mode == "ERASE":
                _draw_cx, _draw_cy = 0, 0
                pinch_detector.update(False)

                # Feature 2: Canvas Pan -- open palm moving sideways = pan
                wrist_x = lm_px(landmarks, 0, fw, fh)[0]
                if pan_prev_wrist_x is not None:
                    _raw_dx, pan_smooth_dx = calculate_dx(
                        pan_prev_wrist_x, wrist_x, pan_smooth_dx)
                    _is_panning = abs(pan_smooth_dx) > PAN_THRESHOLD
                else:
                    _is_panning = False
                pan_prev_wrist_x = wrist_x

                if _is_panning:
                    # Shift canvas horizontally
                    base_canvas = shift_canvas(base_canvas, int(pan_smooth_dx))
                    cv2.putText(frame, "PAN MODE",
                                (fw // 2 - 70, fh // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 255, 200), 2, cv2.LINE_AA)
                    _arrow = ">>" if pan_smooth_dx > 0 else "<<"
                    cv2.putText(frame, _arrow,
                                (fw // 2 - 30, fh // 2 + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                (0, 255, 200), 2, cv2.LINE_AA)
                else:
                    # Normal erase
                    palm_pts = [lm_px(landmarks, i, fw, fh)
                                for i in [0, 4, 8, 12, 16, 20]]
                    pcx = int(np.mean([p[0] for p in palm_pts]))
                    pcy = int(np.mean([p[1] for p in palm_pts]))
                    ccx, ccy = screen_to_canvas(
                        pcx, pcy, fw, fh, cw, ch,
                        offset_x, offset_y, scale_factor)
                    er = max(1, int(ERASER_RADIUS / scale_factor))
                    cv2.circle(base_canvas, (ccx, ccy), er, (0, 0, 0), -1)
                    cv2.circle(frame, (pcx, pcy), ERASER_RADIUS, (0, 100, 255), 2)
                    cv2.putText(frame, "ERASING",
                                (pcx - 35, pcy - ERASER_RADIUS - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                # Finish any active draw stroke before erasing
                if stroke_mgr.active_stroke is not None:
                    s = stroke_mgr.finish_draw()
                    if s:
                        stroke_snap()
                prev_x, prev_y     = None, None
                smooth_x, smooth_y = None, None   # reset so DRAW re-entry snaps to finger
                in_pinch = False
                # Erase nearby 3D strokes whose projected centroid is in range
                if depth_enabled and strokes_3d:
                    proj_all = [
                        project_3D_to_2D(s.points, R_cam, t_cam,
                                         FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                                         cx_cam, cy_cam)
                        for s in strokes_3d
                    ]
                    keep = []
                    for s_idx, stroke in enumerate(strokes_3d):
                        pts_p = [p for p in proj_all[s_idx] if p is not None]
                        if not pts_p:
                            keep.append(stroke)
                            continue
                        cx_s = int(np.mean([p[0] for p in pts_p]))
                        cy_s = int(np.mean([p[1] for p in pts_p]))
                        if np.hypot(cx_s - pcx, cy_s - pcy) > ERASER_RADIUS * 1.5:
                            keep.append(stroke)
                    strokes_3d = keep
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                current_stroke = None

            # ── MOVE ──────────────────────────────────────────
            elif mode == "MOVE":
                _draw_cx, _draw_cy = 0, 0
                pinch_action = pinch_detector.update(True)

                # Finish any active draw stroke
                if stroke_mgr.active_stroke is not None:
                    s = stroke_mgr.finish_draw()
                    if s:
                        stroke_snap()
                        stroke_mgr.render(base_canvas)

                mid_x = int((smooth_x + smooth_thumb_x) / 2)
                mid_y = int((smooth_y + smooth_thumb_y) / 2)

                # Map pinch midpoint to canvas space for object selection/drag
                mid_cx, mid_cy = screen_to_canvas(
                    mid_x, mid_y, fw, fh, cw, ch,
                    offset_x, offset_y, scale_factor)

                if pinch_detector.is_move:
                    if not obj_drag_started:
                        # First drag frame — try to select a stroke object
                        stroke_mgr.try_select(mid_cx, mid_cy)
                        if stroke_mgr.selected_idx >= 0:
                            # Object selected — drag it
                            stroke_mgr.begin_drag(mid_cx, mid_cy)
                            obj_drag_active  = True
                        else:
                            # No object nearby — fall back to canvas pan
                            pinch_anchor_x     = mid_x
                            pinch_anchor_y     = mid_y
                            pinch_anchor_off_x = offset_x
                            pinch_anchor_off_y = offset_y
                            in_pinch           = True
                            obj_drag_active    = False
                        obj_drag_started = True

                    elif obj_drag_active and stroke_mgr.selected_idx >= 0:
                        # Drag selected stroke object
                        stroke_mgr.update_drag(mid_cx, mid_cy)
                        stroke_mgr.render(base_canvas)

                        # ── Drag-to-delete: show trash + check hover ──────
                        drag_trash_visible  = True
                        drag_trash_hovering = is_hovering_trash(mid_x, mid_y, fw)
                        drag_trash_obj_id   = id(stroke_mgr.strokes[stroke_mgr.selected_idx])

                        # Show selection highlight on frame
                        s = stroke_mgr.strokes[stroke_mgr.selected_idx]
                        x1, y1, x2, y2 = s.bounding_box()
                        box_col = (0, 60, 255) if drag_trash_hovering else (0, 255, 200)
                        cv2.rectangle(frame,
                                      (x1 - 8, y1 - 8), (x2 + 8, y2 + 8),
                                      box_col, 2)
                        lbl_drag = "RELEASE TO DELETE!" if drag_trash_hovering else "DRAGGING"
                        cv2.putText(frame, lbl_drag,
                                    (x1, y1 - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    box_col, 2, cv2.LINE_AA)
                    else:
                        # Canvas pan (no object selected)
                        dx = mid_x - pinch_anchor_x
                        dy = mid_y - pinch_anchor_y
                        offset_x = pinch_anchor_off_x + dx
                        offset_y = pinch_anchor_off_y + dy
                else:
                    # Hold threshold not yet crossed
                    in_pinch         = False
                    obj_drag_started = False
                    drag_trash_visible  = False
                    drag_trash_hovering = False

                # Visual feedback
                cv2.circle(frame, (mid_x, mid_y), 14, (255, 100, 0), -1)
                cv2.circle(frame, (mid_x, mid_y), 14, (255, 200, 100), 2)
                cv2.line(frame,
                         (int(smooth_thumb_x), int(smooth_thumb_y)),
                         (ix, iy), (255, 100, 0), 2)
                prev_x, prev_y     = None, None
                smooth_x, smooth_y = None, None
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                current_stroke = None

            # ── DRAW ──────────────────────────────────────────
            elif mode == "DRAW":
                in_pinch = False
                _draw_cx, _draw_cy = 0, 0
                pinch_detector.update(False)   # not pinching (MOVE only)
                obj_drag_active  = False
                obj_drag_started = False

                if is_over_panel(ix, iy):
                    # Hovering over floating panel — check trash bin first, then color/brush
                    if is_over_trash(ix, iy, fw):
                        # Trash bin hover — dwell to delete
                        if not trash_hovering:
                            trash_hovering    = True
                            trash_dwell_start = time.time()
                        else:
                            dwell_elapsed = time.time() - trash_dwell_start
                            if dwell_elapsed >= TRASH_DWELL_SECS:
                                if stroke_mgr.selected_idx >= 0:
                                    stroke_snap()
                                    stroke_mgr.delete_selected()
                                    stroke_mgr.render(base_canvas)
                                    print("[INFO] Selected stroke deleted.")
                                elif stroke_mgr.strokes:
                                    stroke_snap()
                                    stroke_mgr.strokes.pop()
                                    stroke_mgr.render(base_canvas)
                                    print("[INFO] Last stroke deleted.")
                                trash_hovering    = False
                                trash_dwell_start = 0.0
                    else:
                        trash_hovering    = False
                        trash_dwell_start = 0.0
                        # Color / brush selection
                        current_color_idx, current_brush_idx = check_ui_click(
                            ix, iy, current_color_idx, current_brush_idx)
                        # Action buttons via gesture (instant on hover)
                        _act = check_action_click(ix, iy)
                        if _act == 'UNDO':
                            stroke_undo()
                            base_canvas, strokes_3d = undo_manager.undo(
                                base_canvas, strokes_3d)
                        elif _act == 'REDO':
                            stroke_redo()
                            base_canvas, strokes_3d = undo_manager.redo(
                                base_canvas, strokes_3d)
                        elif _act == 'CLEAR':
                            clear_canvas(base_canvas)
                            stroke_mgr.clear()
                            stroke_undo_stack.clear(); stroke_redo_stack.clear()
                            strokes_3d = []; current_stroke = None
                            z_history.clear()
                            offset_x, offset_y = 0, 0
                            scale_factor = 1.0; zoom_target_scale = 1.0
                            single_debouncer.reset(); zoom_debouncer.reset()
                            pinch_detector.reset(); two_finger_detector.reset()
                            undo_manager.reset(); shape_detector.reset()
                            pan_prev_wrist_x = None; pan_smooth_dx = 0.0
                            obj_drag_active = False; obj_drag_started = False
                        elif _act == 'SAVE':
                            fname = f"drawing_{int(time.time())}.png"
                            cv2.imwrite(fname, base_canvas)
                            print(f"[INFO] Saved '{fname}' via gesture.")

                    # End any active stroke when entering UI bar
                    if stroke_mgr.active_stroke is not None:
                        s = stroke_mgr.finish_draw()
                        if s:
                            stroke_snap()
                            stroke_mgr.render(base_canvas)
                    prev_x, prev_y = None, None
                    if current_stroke is not None and len(current_stroke) > 0:
                        strokes_3d.append(current_stroke)
                    current_stroke = None
                else:
                    # Map smoothed screen position → canvas space
                    cx2d, cy2d = screen_to_canvas(
                        ix, iy, fw, fh, cw, ch,
                        offset_x, offset_y, scale_factor)
                    cx2d = int(np.clip(cx2d, 0, cw - 1))
                    cy2d = int(np.clip(cy2d, 0, ch - 1))
                    _draw_cx, _draw_cy = cx2d, cy2d

                    color = COLORS[COLOR_NAMES[current_color_idx]]
                    brush = max(1, int(BRUSH_SIZES[current_brush_idx]
                                       / scale_factor))

                    # ── Feature 1: Stroke reset — no jump on re-entry ─────
                    # If this is the first DRAW frame (prev_x is None),
                    # start a fresh stroke — never connect to old position.
                    if prev_x is None:
                        # Start new stroke object
                        stroke_mgr.begin_draw(color, brush)
                        stroke_mgr.add_draw_point(cx2d, cy2d)
                        # Place a single dot (no line yet)
                        cv2.circle(base_canvas, (cx2d, cy2d),
                                   max(brush // 2, 1), color, -1,
                                   lineType=cv2.LINE_AA)
                        prev_x, prev_y = cx2d, cy2d
                    else:
                        # Continue existing stroke
                        if stroke_mgr.active_stroke is None:
                            stroke_mgr.begin_draw(color, brush)
                        added = stroke_mgr.add_draw_point(cx2d, cy2d)
                        if added and np.hypot(cx2d - prev_x, cy2d - prev_y) >= MIN_DRAW_DIST:
                            cv2.line(base_canvas,
                                     (prev_x, prev_y), (cx2d, cy2d),
                                     color, brush, lineType=cv2.LINE_AA)
                            prev_x, prev_y = cx2d, cy2d

                    # ── 3D stroke accumulation (unchanged) ────────────────
                    if depth_enabled and smoothed_depth is not None:
                        raw_z = depth_smoother.get_depth_at(ix, iy)
                        z_history.append(raw_z)
                        smooth_z = float(np.mean(z_history))
                        X3, Y3, Z3 = convert_to_3D(
                            ix, iy, smooth_z,
                            FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                            cx_cam, cy_cam)
                        p_cam   = np.array([X3, Y3, Z3], dtype=np.float64)
                        p_world = R_cam.T @ (p_cam - t_cam)
                        if current_stroke is None:
                            current_stroke = Stroke3D(color, brush)
                        if len(current_stroke) == 0:
                            current_stroke.add_point(*p_world)
                        else:
                            last = current_stroke.points[-1]
                            if np.linalg.norm(np.array(p_world) - np.array(last)) > 0.001:
                                current_stroke.add_point(*p_world)

                cv2.circle(frame, (ix, iy), 7,
                           COLORS[COLOR_NAMES[current_color_idx]], -1)
                cv2.circle(frame, (ix, iy), 7, (255, 255, 255), 1)

            # ── CURSOR ────────────────────────────────────────
            # Index + Middle up = CURSOR mode
            # Tap both fingers down quickly = UNDO
            # Double-tap = REDO
            elif mode == "CURSOR":
                _draw_cx, _draw_cy = 0, 0
                in_pinch = False
                pinch_detector.update(False)   # not pinching

                # Two-finger tap detection for UNDO / REDO
                tap_action = two_finger_detector.update(landmarks)
                if tap_action == "UNDO":
                    stroke_undo()
                    base_canvas, strokes_3d = undo_manager.undo(
                        base_canvas, strokes_3d)
                elif tap_action == "REDO":
                    stroke_redo()
                    base_canvas, strokes_3d = undo_manager.redo(
                        base_canvas, strokes_3d)

                prev_x, prev_y     = None, None
                smooth_x, smooth_y = None, None
                trash_hovering    = False
                trash_dwell_start = 0.0

                # Visual: hollow circle cursor + tap indicator
                cursor_col = (255, 200, 0)
                if two_finger_detector._down:
                    cursor_col = (100, 255, 100)   # green flash when dipped
                cv2.circle(frame, (ix, iy), 14, cursor_col, 2)
                cv2.circle(frame, (ix, iy), 4,  cursor_col, -1)
                # Show tap hint
                cv2.putText(frame, "TAP=UNDO  DBL=REDO",
                            (10, fh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 100), 1, cv2.LINE_AA)

                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                    undo_manager.snapshot(base_canvas, strokes_3d)
                current_stroke = None

            # ── IDLE ──────────────────────────────────────────
            else:
                _draw_cx, _draw_cy = 0, 0
                in_pinch = False
                smooth_x, smooth_y = None, None
                trash_hovering    = False
                trash_dwell_start = 0.0
                pinch_detector.update(False)
                prev_x, prev_y = None, None
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                    undo_manager.snapshot(base_canvas, strokes_3d)
                current_stroke = None

            # ── Shape detector ────────────────────────────────
            _sd_cx  = _draw_cx if mode == "DRAW" else 0
            _sd_cy  = _draw_cy if mode == "DRAW" else 0
            _was_drawing_before = shape_detector.was_drawing
            shape_name, new_points = shape_detector.update(
                mode, _sd_cx, _sd_cy, cw, ch)
            
            # Snapshot when a 2D stroke just completed (shape detector fired)
            if _was_drawing_before and not shape_detector.was_drawing:
                if stroke_mgr.active_stroke is not None:
                    s = stroke_mgr.finish_draw()
                    if s:
                        if new_points and stroke_mgr.strokes:
                            stroke_mgr.strokes[-1].points = new_points
                        stroke_mgr.render(base_canvas)
                        stroke_snap()
                undo_manager.snapshot(base_canvas, strokes_3d)

            if mode != "MOVE" and obj_drag_active:
                # Drag ended — check if dropped on trash bin
                if drag_trash_visible and drag_trash_hovering and stroke_mgr.selected_idx >= 0:
                    stroke_snap()
                    deleted = delete_stroke(stroke_mgr, drag_trash_obj_id)
                    if deleted:
                        stroke_mgr.render(base_canvas)
                        print("[INFO] Stroke deleted by drag-to-trash.")
                    # Flash feedback
                    cv2.putText(frame, "DELETED!",
                                (fw // 2 - 60, fh // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 60, 255), 3, cv2.LINE_AA)
                else:
                    stroke_mgr.end_drag()
                    stroke_snap()
                drag_trash_visible  = False
                drag_trash_hovering = False
                drag_trash_obj_id   = -1
                obj_drag_active  = False
                obj_drag_started = False

        # ══════════════════════════════════════════════════════
        # NO HANDS
        # ══════════════════════════════════════════════════════
        else:
            # Flush clap state when hands leave frame
            clap_detector.update([], fw, fh)
            prev_x, prev_y     = None, None
            smooth_x, smooth_y = None, None
            in_pinch           = False
            in_zoom            = False
            zoom_initial_dist  = None
            single_debouncer.reset()
            zoom_debouncer.reset()
            pinch_detector.reset()
            pan_prev_wrist_x = None   # reset pan anchor when hand leaves
            if current_stroke is not None and len(current_stroke) > 0:
                strokes_3d.append(current_stroke)
                undo_manager.snapshot(base_canvas, strokes_3d)
            current_stroke = None
            shape_detector.update("IDLE", 0, 0, cw, ch)

        # ── Render 2D canvas ──────────────────────────────────────────────
        frame = render_canvas(frame, base_canvas,
                              offset_x, offset_y, scale_factor)

        # ── Drag-to-delete trash bin (shown only while dragging a stroke) ─
        if drag_trash_visible:
            draw_trash_icon(frame, drag_trash_hovering)

        # ── Feature 1: Object detection + Blueprint overlay ───────────────
        if detections:
            if blueprint_mode:
                for det in detections:
                    roi = extract_roi(frame, det)
                    if roi is not None:
                        bp = apply_blueprint_effect(roi)
                        frame = overlay_blueprint(frame, bp, det)
            else:
                frame = draw_detections(frame, detections)

        # ── Render 3D strokes on top ──────────────────────────────────────
        if depth_enabled and strokes_3d:
            render_3D_lines(frame, strokes_3d,
                            R_cam, t_cam,
                            FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                            cx_cam, cy_cam)
        if depth_enabled and current_stroke is not None and len(current_stroke) >= 2:
            render_3D_lines(frame, [current_stroke],
                            R_cam, t_cam,
                            FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                            cx_cam, cy_cam)

        # ── Depth map mini-overlay (bottom-right corner) ──────────────────
        if depth_enabled and smoothed_depth is not None:
            dm_vis   = (smoothed_depth * 255).astype(np.uint8)
            dm_col   = cv2.applyColorMap(dm_vis, cv2.COLORMAP_INFERNO)
            ow, oh   = 200, 112
            dm_small = cv2.resize(dm_col, (ow, oh))
            frame[fh - oh - 10: fh - 10, fw - ow - 10: fw - 10] = dm_small
            cv2.rectangle(frame,
                          (fw - ow - 10, fh - oh - 10),
                          (fw - 10, fh - 10),
                          (100, 100, 100), 1)
            cv2.putText(frame, "Depth", (fw - ow - 5, fh - oh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ── UI bar ────────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Combine action labels: pinch > two-finger > clap
        _action_lbl = (pinch_detector.action_label or
                       two_finger_detector.action_label or
                       clap_detector.action_label)

        # ── Clap visual flash ─────────────────────────────────────────────
        # Show a bright full-screen overlay for 0.4s after a clap
        if clap_detector.action_label:
            _flash_age = time.time() - clap_detector._action_label_time
            if _flash_age < 0.4:
                _alpha = max(0.0, 0.5 * (1.0 - _flash_age / 0.4))
                _overlay = frame.copy()
                cv2.rectangle(_overlay, (0, 0), (fw, fh), (0, 255, 120), -1)
                cv2.addWeighted(_overlay, _alpha, frame, 1 - _alpha, 0, frame)
            # Large centred text
            _txt = "CLAP DETECTED - CANVAS CLEARED"
            (_tw, _th), _ = cv2.getTextSize(
                _txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.rectangle(frame,
                          (fw // 2 - _tw // 2 - 16, fh // 2 - _th - 16),
                          (fw // 2 + _tw // 2 + 16, fh // 2 + 16),
                          (0, 60, 0), -1)
            cv2.putText(frame, _txt,
                        (fw // 2 - _tw // 2, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 120), 3, cv2.LINE_AA)

        _trash_prog = 0.0
        if trash_hovering and trash_dwell_start > 0:
            _trash_prog = min(1.0, (time.time() - trash_dwell_start) / TRASH_DWELL_SECS)
        frame = draw_ui(frame, current_color_idx, current_brush_idx,
                        mode, fps, scale_factor,
                        shape_label=shape_detector.label,
                        depth_enabled=depth_enabled,
                        action_label=_action_lbl,
                        trash_hover_progress=_trash_prog,
                        trash_has_selection=(stroke_mgr.selected_idx >= 0))

        cv2.imshow(
            "Hand Gesture Drawing  [d=3D  c=clear  s=save  q=quit]", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quit.")
            break

        elif key == ord('o'):
            # Toggle object detection (YOLOv8)
            obj_detector.toggle()

        elif key == ord('b'):
            # Toggle blueprint mode (requires object detection to be ON)
            if not obj_detector.enabled:
                obj_detector.enable()
            blueprint_mode = not blueprint_mode
            print(f"[INFO] Blueprint mode {'ON' if blueprint_mode else 'OFF'}.")

        elif key == ord('d'):
            # Toggle depth mode; load MiDaS on first activation
            if not depth_enabled:
                if not depth_ready:
                    depth_ready = depth_estimator.load()
                if depth_ready:
                    depth_enabled = True
                    print("[INFO] 3D depth mode ON.")
                else:
                    print("[WARN] Could not enable depth mode "
                          "(MiDaS failed to load).")
            else:
                depth_enabled = False
                print("[INFO] 3D depth mode OFF.")

        elif key == ord('c'):
            clear_canvas(base_canvas)
            stroke_mgr.clear()
            stroke_undo_stack.clear()
            stroke_redo_stack.clear()
            strokes_3d     = []
            current_stroke = None
            z_history.clear()
            offset_x, offset_y = 0, 0
            scale_factor        = 1.0
            zoom_target_scale   = 1.0
            single_debouncer.reset()
            zoom_debouncer.reset()
            pinch_detector.reset()
            two_finger_detector.reset()
            clap_detector.reset()
            undo_manager.reset()
            shape_detector.reset()
            pan_prev_wrist_x = None
            pan_smooth_dx    = 0.0
            obj_drag_active  = False
            obj_drag_started = False
            drag_trash_visible  = False
            drag_trash_hovering = False
            drag_trash_obj_id   = -1
            print("[INFO] Canvas cleared.")

        elif key == ord('s'):
            fname = f"drawing_{int(time.time())}.png"
            cv2.imwrite(fname, base_canvas)
            print(f"[INFO] Saved '{fname}'")
            if depth_enabled and strokes_3d:
                fname3d = f"drawing_3d_{int(time.time())}.png"
                cv2.imwrite(fname3d, frame)
                print(f"[INFO] 3D composite saved '{fname3d}'")

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()


if __name__ == "__main__":
    main()
