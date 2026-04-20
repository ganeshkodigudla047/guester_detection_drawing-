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
  - Pinch (thumb+index)   → MOVE MODE

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
YOLO_CONF          = 0.45           # minimum detection confidence
YOLO_INPUT_SIZE    = 416            # resize frame to this before YOLO inference
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

# UI layout
COLOR_BTN_W      = 80
COLOR_BTN_H      = 40
COLOR_BTN_Y      = 10
COLOR_BTN_MARGIN = 10

BRUSH_SIZES      = [3, 6, 10, 16]
BRUSH_BTN_X_START = 720
BRUSH_BTN_Y      = 10
BRUSH_BTN_W      = 50
BRUSH_BTN_H      = 40

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
    def update(self, mode, cx, cy, base_canvas, color, brush,
               cw, ch):
        """
        Call once per frame from the main loop.

        Parameters
        ----------
        mode         : current debounced mode string
        cx, cy       : canvas-space finger position (only used when mode==DRAW)
        base_canvas  : the drawing surface (modified in-place on conversion)
        color        : current drawing color (BGR tuple)
        brush        : current brush thickness
        cw, ch       : canvas width / height (for bounds clamping)

        Returns
        -------
        label        : shape name to display (empty string = nothing to show)
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
            detected = self._detect_and_replace(base_canvas, color, brush, cw, ch)
            self.points = []
            if detected:
                self.label       = detected
                self.label_timer = time.time()
                print(f"[SHAPE] Detected: {detected}")

        else:
            # Not drawing and wasn't drawing — nothing to do
            self.was_drawing = False

        # Clear label after display timeout
        if self.label and (time.time() - self.label_timer > SHAPE_LABEL_SECS):
            self.label = ""

        return self.label

    # ── internal: classify + replace ─────────
    def _detect_and_replace(self, canvas, color, brush, cw, ch):
        """
        Analyse self.points, classify the shape, erase the rough stroke
        from the bounding region, and draw a clean geometric shape.
        Returns the shape name string, or None if detection failed.
        """
        if len(self.points) < SHAPE_MIN_POINTS:
            return None

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
            return None

        # ── 4. Approximate polygon ────────────────────────────────────
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, SHAPE_POLY_EPSILON * peri, True)
        n      = len(approx)

        # ── 5. Classify ───────────────────────────────────────────────
        if n == 2:
            shape_name = "LINE"
        elif n == 3:
            shape_name = "TRIANGLE"
        elif n == 4:
            # Distinguish square vs rectangle by aspect ratio
            aspect = w / float(h) if h > 0 else 1.0
            shape_name = "SQUARE" if 0.85 <= aspect <= 1.15 else "RECTANGLE"
        elif n > 6:
            # Extra circularity check: area / (pi * r^2)
            area      = cv2.contourArea(contour)
            (cx2, cy2), radius = cv2.minEnclosingCircle(contour)
            if radius > 0:
                circularity = area / (np.pi * radius ** 2)
            else:
                circularity = 0.0
            shape_name = "CIRCLE" if circularity > 0.65 else "POLYGON"
        else:
            shape_name = "POLYGON"

        # ── 6. Erase the rough stroke from the bounding region ────────
        # Add a small margin so we don't leave artefacts at the edges
        margin = max(brush + 4, 8)
        ex1 = max(0,  x - margin)
        ey1 = max(0,  y - margin)
        ex2 = min(cw, x + w + margin)
        ey2 = min(ch, y + h + margin)

        # Build a mask of the rough stroke pixels and erase only those
        stroke_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.polylines(stroke_mask, [pts_smooth.reshape((-1, 1, 2))],
                      False, 255, brush + 2)
        cv2.dilate(stroke_mask, np.ones((5, 5), np.uint8), dst=stroke_mask)
        canvas[stroke_mask > 0] = 0

        # ── 7. Draw clean shape ───────────────────────────────────────
        thickness = max(brush, 2)

        if shape_name == "LINE":
            start = tuple(pts_smooth[0])
            end   = tuple(pts_smooth[-1])
            cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)

        elif shape_name == "TRIANGLE":
            cv2.drawContours(canvas, [approx], -1, color,
                             thickness, lineType=cv2.LINE_AA)

        elif shape_name in ("RECTANGLE", "SQUARE"):
            cv2.rectangle(canvas, (x, y), (x + w, y + h),
                          color, thickness, lineType=cv2.LINE_AA)

        elif shape_name == "CIRCLE":
            (cx2, cy2), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(canvas, (int(cx2), int(cy2)), int(radius),
                       color, thickness, lineType=cv2.LINE_AA)

        else:  # POLYGON — draw the approximated contour as-is
            cv2.drawContours(canvas, [approx], -1, color,
                             thickness, lineType=cv2.LINE_AA)

        return shape_name

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
    State machine that watches pinch on/off transitions and
    classifies them as MOVE, UNDO, or REDO.

    Call update(is_pinching_now) every frame.
    It returns an action string when one fires, else None.
    """
    def __init__(self):
        self._active       = False   # pinch is currently held
        self._start_time   = 0.0     # when current pinch started
        self._tap_count    = 0       # taps in current burst
        self._last_tap_end = 0.0     # when the last tap released
        self._last_action  = 0.0     # time of last fired action (cooldown)
        self.is_move       = False   # True while MOVE (hold) is active

        # For UI feedback
        self.action_label       = ""
        self._action_label_time = 0.0

    def update(self, pinching_now):
        """
        Feed the current pinch state each frame.

        Parameters
        ----------
        pinching_now : bool — is the hand currently pinching?

        Returns
        -------
        action : str or None — 'MOVE', 'UNDO', 'REDO', or None
        """
        now    = time.time()
        action = None

        # ── Pinch just started ────────────────────────────────
        if pinching_now and not self._active:
            self._active     = True
            self._start_time = now

        # ── Pinch is being held ───────────────────────────────
        elif pinching_now and self._active:
            hold_dur = now - self._start_time
            if hold_dur >= PINCH_HOLD_SECS and not self.is_move:
                # Crossed the hold threshold → activate MOVE
                self.is_move = True
                self._tap_count = 0   # cancel any pending tap count
                action = "MOVE"
                self._set_label("MOVE MODE")

        # ── Pinch just released ───────────────────────────────
        elif not pinching_now and self._active:
            hold_dur = now - self._start_time
            self._active = False

            if self.is_move:
                # End of a MOVE drag — no tap action
                self.is_move = False
            elif hold_dur < PINCH_TAP_MAX_SECS:
                # Quick tap — check for double-tap window
                if (self._tap_count == 1 and
                        now - self._last_tap_end < PINCH_DOUBLE_SECS):
                    # Second tap within window → REDO
                    if now - self._last_action >= PINCH_COOLDOWN:
                        action = "REDO"
                        self._set_label("REDO")
                        self._last_action = now
                    self._tap_count = 0
                else:
                    # First tap — start counting
                    self._tap_count  = 1
                    self._last_tap_end = now

            self._start_time = 0.0

        # ── Check if single-tap window has expired → UNDO ─────
        if (not pinching_now and
                self._tap_count == 1 and
                now - self._last_tap_end >= PINCH_DOUBLE_SECS):
            if now - self._last_action >= PINCH_COOLDOWN:
                action = "UNDO"
                self._set_label("UNDO")
                self._last_action = now
            self._tap_count = 0

        # ── Clear label after timeout ─────────────────────────
        if (self.action_label and
                now - self._action_label_time > ACTION_LABEL_SECS):
            self.action_label = ""

        return action

    def _set_label(self, text):
        self.action_label       = text
        self._action_label_time = time.time()

    def reset(self):
        self._active     = False
        self._start_time = 0.0
        self._tap_count  = 0
        self._last_tap_end = 0.0
        self.is_move     = False
        self.action_label = ""


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
            depth_enabled=False, action_label=""):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # Color buttons
    for i, name in enumerate(COLOR_NAMES):
        bx = COLOR_BTN_MARGIN + i * (COLOR_BTN_W + COLOR_BTN_MARGIN)
        by = COLOR_BTN_Y
        cv2.rectangle(frame, (bx, by),
                      (bx + COLOR_BTN_W, by + COLOR_BTN_H),
                      COLORS[name], -1)
        if i == current_color_idx:
            cv2.rectangle(frame, (bx - 2, by - 2),
                          (bx + COLOR_BTN_W + 2, by + COLOR_BTN_H + 2),
                          (255, 255, 255), 2)
        cv2.putText(frame, name[:3], (bx + 5, by + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Brush size buttons
    for i, size in enumerate(BRUSH_SIZES):
        bx = BRUSH_BTN_X_START + i * (BRUSH_BTN_W + COLOR_BTN_MARGIN)
        by = BRUSH_BTN_Y
        cv2.rectangle(frame, (bx, by),
                      (bx + BRUSH_BTN_W, by + BRUSH_BTN_H),
                      (80, 80, 80), -1)
        if i == current_brush_idx:
            cv2.rectangle(frame, (bx - 2, by - 2),
                          (bx + BRUSH_BTN_W + 2, by + BRUSH_BTN_H + 2),
                          (255, 255, 255), 2)
        cv2.circle(frame,
                   (bx + BRUSH_BTN_W // 2, by + BRUSH_BTN_H // 2),
                   size, (200, 200, 200), -1)

    # Mode label
    mode_colors = {
        "DRAW":   (0, 255, 100),
        "CURSOR": (255, 200, 0),
        "ERASE":  (0, 100, 255),
        "MOVE":   (255, 100, 0),
        "ZOOM":   (0, 220, 255),
        "IDLE":   (150, 150, 150),
    }
    label = f"MODE: {mode}"
    if mode == "ERASE":
        label = "ERASER MODE (PALM)"
    elif mode == "ZOOM":
        label = f"ZOOM MODE  {int(scale_factor * 100)}%"
    mc = mode_colors.get(mode, (200, 200, 200))
    cv2.putText(frame, label, (w - 420, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, mc, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Depth mode indicator
    if depth_enabled:
        cv2.putText(frame, "3D DEPTH ON", (w - 230, 58 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

    # Action label (UNDO / REDO / MOVE MODE) — large centred banner
    if action_label:
        label_colors = {
            "UNDO":      (80,  80,  255),
            "REDO":      (80,  200, 80),
            "MOVE MODE": (255, 140, 0),
        }
        ac = label_colors.get(action_label, (220, 220, 220))
        (tw, th), _ = cv2.getTextSize(action_label,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        lx = (w - tw) // 2
        ly = h // 2
        cv2.rectangle(frame,
                      (lx - 20, ly - th - 16),
                      (lx + tw + 20, ly + 16),
                      (15, 15, 15), -1)
        cv2.rectangle(frame,
                      (lx - 20, ly - th - 16),
                      (lx + tw + 20, ly + 16),
                      ac, 3)
        cv2.putText(frame, action_label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, ac, 4,
                    lineType=cv2.LINE_AA)

    # Shape detection label — shown below the UI bar when active
    if shape_label:
        label_text = f"Shape: {shape_label}"
        (tw, th), _ = cv2.getTextSize(label_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        lx = (w - tw) // 2
        ly = 105
        # Dark pill background
        cv2.rectangle(frame, (lx - 14, ly - th - 10),
                      (lx + tw + 14, ly + 10),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (lx - 14, ly - th - 10),
                      (lx + tw + 14, ly + 10),
                      (0, 220, 120), 2)
        cv2.putText(frame, label_text, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 120), 3,
                    lineType=cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# check_ui_click()
# ─────────────────────────────────────────────
def check_ui_click(ix, iy, current_color_idx, current_brush_idx):
    if COLOR_BTN_Y <= iy <= COLOR_BTN_Y + COLOR_BTN_H:
        for i in range(len(COLOR_NAMES)):
            bx = COLOR_BTN_MARGIN + i * (COLOR_BTN_W + COLOR_BTN_MARGIN)
            if bx <= ix <= bx + COLOR_BTN_W:
                return i, current_brush_idx

    if BRUSH_BTN_Y <= iy <= BRUSH_BTN_Y + BRUSH_BTN_H:
        for i in range(len(BRUSH_SIZES)):
            bx = BRUSH_BTN_X_START + i * (BRUSH_BTN_W + COLOR_BTN_MARGIN)
            if bx <= ix <= bx + BRUSH_BTN_W:
                return current_color_idx, i

    return current_color_idx, current_brush_idx

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
        # Resize to YOLO_INPUT_SIZE for speed, run inference
        scale_x = fw / YOLO_INPUT_SIZE
        scale_y = fh / YOLO_INPUT_SIZE
        small   = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

        results = self._model(small, conf=YOLO_CONF, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Scale back to original frame size
                detections.append({
                    'label': r.names[int(box.cls[0])],
                    'conf' : float(box.conf[0]),
                    'x1'   : int(x1 * scale_x),
                    'y1'   : int(y1 * scale_y),
                    'x2'   : int(x2 * scale_x),
                    'y2'   : int(y2 * scale_y),
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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

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
    pinch_detector   = PinchActionDetector()   # tap/hold → UNDO/REDO/MOVE
    undo_manager     = UndoRedoManager()       # canvas + 3D stroke history

    # ── Feature 1: Object detection + Blueprint ───────────────────────────
    obj_detector   = ObjectDetector()
    blueprint_mode = False   # 'b' toggles blueprint rendering
    detections     = []      # last YOLO results (list of dicts)

    # ── Feature 2: Canvas pan ─────────────────────────────────────────────
    pan_prev_wrist_x = None   # wrist x from previous frame (screen px)
    pan_smooth_dx    = 0.0    # EMA-smoothed horizontal delta

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

            if stable_two == "ZOOM":
                mode = "ZOOM"
                single_debouncer.reset()
                (in_zoom, zoom_initial_dist, zoom_initial_scale,
                 zoom_target_scale, scale_factor,
                 pt1, pt2) = zoom_canvas(
                    lms_list[0], lms_list[1], fw, fh,
                    in_zoom, zoom_initial_dist, zoom_initial_scale,
                    zoom_target_scale, scale_factor)
                cv2.line(frame, pt1, pt2, (0, 220, 255), 2)
                cv2.circle(frame, pt1, 12, (0, 220, 255), -1)
                cv2.circle(frame, pt2, 12, (0, 220, 255), -1)
                mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(frame, f"Zoom: {int(scale_factor * 100)}%",
                            (mid[0] - 45, mid[1] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            else:
                in_zoom = False
                zoom_initial_dist = None
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
                prev_x, prev_y = None, None
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

                # Feed pinch state to the tap/hold detector every frame
                pinch_action = pinch_detector.update(True)

                # Fire UNDO / REDO if a tap action was returned
                # (shouldn't happen mid-hold, but guard anyway)
                if pinch_action == "UNDO":
                    base_canvas, strokes_3d = undo_manager.undo(
                        base_canvas, strokes_3d)
                elif pinch_action == "REDO":
                    base_canvas, strokes_3d = undo_manager.redo(
                        base_canvas, strokes_3d)

                # Drag logic — only active once hold threshold is crossed
                mid_x = int((smooth_x + smooth_thumb_x) / 2)
                mid_y = int((smooth_y + smooth_thumb_y) / 2)

                if pinch_detector.is_move:
                    if not in_pinch:
                        pinch_anchor_x     = mid_x
                        pinch_anchor_y     = mid_y
                        pinch_anchor_off_x = offset_x
                        pinch_anchor_off_y = offset_y
                        in_pinch = True
                    else:
                        dx = mid_x - pinch_anchor_x
                        dy = mid_y - pinch_anchor_y
                        offset_x = pinch_anchor_off_x + dx
                        offset_y = pinch_anchor_off_y + dy
                else:
                    # Still in tap-detection window — don't drag yet
                    in_pinch = False

                cv2.circle(frame, (mid_x, mid_y), 14, (255, 100, 0), -1)
                cv2.circle(frame, (mid_x, mid_y), 14, (255, 200, 100), 2)
                cv2.line(frame,
                         (int(smooth_thumb_x), int(smooth_thumb_y)),
                         (ix, iy), (255, 100, 0), 2)
                prev_x, prev_y = None, None
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                current_stroke = None

            # ── DRAW ──────────────────────────────────────────
            elif mode == "DRAW":
                in_pinch = False
                _draw_cx, _draw_cy = 0, 0
                pinch_action = pinch_detector.update(False)   # not pinching
                # Tap actions while drawing (finger briefly pinched mid-stroke)
                if pinch_action == "UNDO":
                    base_canvas, strokes_3d = undo_manager.undo(
                        base_canvas, strokes_3d)
                    prev_x, prev_y = None, None
                elif pinch_action == "REDO":
                    base_canvas, strokes_3d = undo_manager.redo(
                        base_canvas, strokes_3d)
                    prev_x, prev_y = None, None

                if iy < 60:
                    # Hovering over UI bar
                    current_color_idx, current_brush_idx = check_ui_click(
                        ix, iy, current_color_idx, current_brush_idx)
                    prev_x, prev_y = None, None
                    if current_stroke is not None and len(current_stroke) > 0:
                        strokes_3d.append(current_stroke)
                    current_stroke = None
                else:
                    # ── 2D canvas draw (always active) ────────
                    cx2d, cy2d = screen_to_canvas(
                        ix, iy, fw, fh, cw, ch,
                        offset_x, offset_y, scale_factor)
                    cx2d = int(np.clip(cx2d, 0, cw - 1))
                    cy2d = int(np.clip(cy2d, 0, ch - 1))
                    _draw_cx, _draw_cy = cx2d, cy2d

                    color = COLORS[COLOR_NAMES[current_color_idx]]
                    brush = max(1, int(BRUSH_SIZES[current_brush_idx]
                                       / scale_factor))

                    if prev_x is not None:
                        if np.hypot(cx2d - prev_x, cy2d - prev_y) >= MIN_DRAW_DIST:
                            cv2.line(base_canvas,
                                     (prev_x, prev_y), (cx2d, cy2d),
                                     color, brush, lineType=cv2.LINE_AA)
                            prev_x, prev_y = cx2d, cy2d
                    else:
                        cv2.circle(base_canvas, (cx2d, cy2d),
                                   max(brush // 2, 1), color, -1,
                                   lineType=cv2.LINE_AA)
                        prev_x, prev_y = cx2d, cy2d

                    # ── 3D stroke accumulation ─────────────────
                    if depth_enabled and smoothed_depth is not None:
                        raw_z = depth_smoother.get_depth_at(ix, iy)
                        z_history.append(raw_z)
                        smooth_z = float(np.mean(z_history))

                        X3, Y3, Z3 = convert_to_3D(
                            ix, iy, smooth_z,
                            FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
                            cx_cam, cy_cam)

                        # Transform to world space: p_world = R^T (p_cam - t)
                        p_cam   = np.array([X3, Y3, Z3], dtype=np.float64)
                        p_world = R_cam.T @ (p_cam - t_cam)

                        if current_stroke is None:
                            current_stroke = Stroke3D(color, brush)

                        if len(current_stroke) == 0:
                            current_stroke.add_point(*p_world)
                        else:
                            last = current_stroke.points[-1]
                            dist3d = np.linalg.norm(
                                np.array(p_world) - np.array(last))
                            if dist3d > 0.001:
                                current_stroke.add_point(*p_world)

                cv2.circle(frame, (ix, iy), 7,
                           COLORS[COLOR_NAMES[current_color_idx]], -1)
                cv2.circle(frame, (ix, iy), 7, (255, 255, 255), 1)

            # ── CURSOR ────────────────────────────────────────
            elif mode == "CURSOR":
                _draw_cx, _draw_cy = 0, 0
                in_pinch = False
                pinch_action = pinch_detector.update(False)
                if pinch_action == "UNDO":
                    base_canvas, strokes_3d = undo_manager.undo(
                        base_canvas, strokes_3d)
                elif pinch_action == "REDO":
                    base_canvas, strokes_3d = undo_manager.redo(
                        base_canvas, strokes_3d)
                prev_x, prev_y = None, None
                cv2.circle(frame, (ix, iy), 12, (255, 200, 0), 2)
                cv2.circle(frame, (ix, iy), 3,  (255, 200, 0), -1)
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                    undo_manager.snapshot(base_canvas, strokes_3d)
                current_stroke = None

            # ── IDLE ──────────────────────────────────────────
            else:
                _draw_cx, _draw_cy = 0, 0
                in_pinch = False
                pinch_action = pinch_detector.update(False)
                if pinch_action == "UNDO":
                    base_canvas, strokes_3d = undo_manager.undo(
                        base_canvas, strokes_3d)
                elif pinch_action == "REDO":
                    base_canvas, strokes_3d = undo_manager.redo(
                        base_canvas, strokes_3d)
                prev_x, prev_y = None, None
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes_3d.append(current_stroke)
                    undo_manager.snapshot(base_canvas, strokes_3d)
                current_stroke = None

            # ── Shape detector ────────────────────────────────
            _sd_cx  = _draw_cx if mode == "DRAW" else 0
            _sd_cy  = _draw_cy if mode == "DRAW" else 0
            _color  = COLORS[COLOR_NAMES[current_color_idx]]
            _brush  = max(1, int(BRUSH_SIZES[current_brush_idx] / scale_factor))
            _was_drawing_before = shape_detector.was_drawing
            shape_detector.update(
                mode, _sd_cx, _sd_cy,
                base_canvas, _color, _brush, cw, ch)
            # Snapshot when a 2D stroke just completed (shape detector fired)
            if _was_drawing_before and not shape_detector.was_drawing:
                undo_manager.snapshot(base_canvas, strokes_3d)

        # ══════════════════════════════════════════════════════
        # NO HANDS
        # ══════════════════════════════════════════════════════
        else:
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
            shape_detector.update(
                "IDLE", 0, 0, base_canvas,
                COLORS[COLOR_NAMES[current_color_idx]],
                max(1, int(BRUSH_SIZES[current_brush_idx] / scale_factor)),
                cw, ch)

        # ── Render 2D canvas ──────────────────────────────────────────────
        frame = render_canvas(frame, base_canvas,
                              offset_x, offset_y, scale_factor)

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
        frame = draw_ui(frame, current_color_idx, current_brush_idx,
                        mode, fps, scale_factor,
                        shape_label=shape_detector.label,
                        depth_enabled=depth_enabled,
                        action_label=pinch_detector.action_label)

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
            base_canvas[:] = 0
            strokes_3d     = []
            current_stroke = None
            z_history.clear()
            offset_x, offset_y = 0, 0
            scale_factor        = 1.0
            zoom_target_scale   = 1.0
            single_debouncer.reset()
            zoom_debouncer.reset()
            pinch_detector.reset()
            undo_manager.reset()
            shape_detector.reset()
            pan_prev_wrist_x = None
            pan_smooth_dx    = 0.0
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
