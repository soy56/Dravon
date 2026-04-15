"""
main.py — OrbitX Task 4: Dynamic Humanoid Walking Simulation
=============================================================
Integrates MediaPipe webcam-based pose estimation with the Angad humanoid
robot model in MuJoCo for real-time human-to-robot joint mapping.

Pipeline:
  1. Webcam → MediaPipe PoseLandmarker (33 landmarks, Tasks API)
  2. Pose landmarks → 3D joint angle extraction (BodyCapture module)
  3. Joint angles → Constraint handling (URDF limits + hip axis factor)
  4. Constrained angles → Adaptive EMA smoothing (deadband + glitch rejection)
  5. Smoothed angles → MuJoCo position actuators (Angad humanoid, 23 DOF)
  6. MuJoCo viewer displays the robot mirroring human poses in real-time

Mapped Joints (23 DOF via BodyCapture):
  Legs (12): hip_pitch, thigh_roll, thigh_yaw, knee_pitch,
             ankle_pitch, ankle_roll (L/R)
  Torso (3): torso_pitch, torso_roll, torso_yaw
  Arms (8):  shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch (L/R)
  Head (2):  neck_pitch, neck_yaw

Sign Conventions (from URDF / IK solver):
  hip_pitch_l:  NEGATIVE = thigh swings FORWARD
  hip_pitch_r:  POSITIVE = thigh swings FORWARD
  knee_pitch:   NEGATIVE = knee BENDS (flexion)
  ankle_pitch:  POSITIVE = foot stays FLAT
  shoulder_pitch: POSITIVE = arm raises

Usage:
  sim_venv\\Scripts\\python.exe main.py
"""

# Suppress C++ diagnostic logging from TensorFlow/MediaPipe BEFORE imports
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import numpy as np
import threading

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

import mujoco
import mujoco.viewer
from body_capture import BodyCapture
from yolo_tracker import YoloTracker
from action_classifier import ActionClassifier
from action_animator import ActionAnimator
from smpl_tracker import SMPLTracker

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Hip pitch axis projection factor from URDF
# The hip_pitch axis is tilted: (-0.94, 0, 0.34) for left
# Only 94% of the commanded angle produces sagittal-plane pitch
HIP_AXIS_FACTOR = 0.939693

# Smoothing factor for EMA filter (lower = smoother but more lag)
SMOOTHING_ALPHA = 0.35

# Minimum MediaPipe landmark visibility to trust (0-1)
MIN_VISIBILITY = 0.5

# ── Walking Gait Parameters ──
WALK_FREQ       = 0.8    # Hz — slow deliberate walk
WALK_HIP_AMP    = 0.35   # rad — hip swing (~20 degrees)
WALK_KNEE_AMP   = 0.50   # rad — knee bend during swing
WALK_ARM_AMP    = 0.25   # rad — arm counter-swing
WALK_ELBOW_BEND = 0.30   # rad — slight elbow bend
WALK_TORSO_LEAN = 0.05   # rad — subtle forward lean
WALK_RAMP_TIME  = 1.0    # seconds — smooth start
WALK_SPEED      = 0.25   # m/s — forward walking speed

# Joint limits [rad] — from URDF, used for safe clamping
# ALL 23 joints must be listed to satisfy Task 4 Requirement #3:
# "Manage mismatches between human and robot joint limits"
JOINT_LIMITS = {
    # Legs (12 joints)
    'hip_pitch_l':      (-1.57, 1.57),
    'hip_pitch_r':      (-1.57, 1.57),
    'thigh_roll_l':     (-0.50, 0.50),
    'thigh_roll_r':     (-0.50, 0.50),
    'thigh_yaw_l':      (-0.50, 0.50),
    'thigh_yaw_r':      (-0.50, 0.50),
    'knee_pitch_l':     (-2.0,  2.0),
    'knee_pitch_r':     (-2.0,  2.0),
    'ankle_pitch_l':    (-1.0,  1.0),
    'ankle_pitch_r':    (-1.0,  1.0),
    'ankle_roll_l':     (-0.50, 0.50),
    'ankle_roll_r':     (-0.50, 0.50),
    # Torso (3 joints)
    'torso_pitch':      (-1.57, 1.57),
    'torso_roll':       (-1.57, 1.57),
    'torso_yaw':        (-1.57, 1.57),
    # Arms (8 joints)
    'shoulder_pitch_l': (-3.14, 3.14),
    'shoulder_pitch_r': (-3.14, 3.14),
    'shoulder_roll_l':  ( 0.00, 2.79),
    'shoulder_roll_r':  (-2.88, 0.17),
    'elbow_yaw_l':      (-3.14, 3.14),
    'elbow_yaw_r':      (-3.14, 3.14),
    'elbow_pitch_l':    (-1.57, 1.57),
    'elbow_pitch_r':    (-1.57, 1.57),
    # Head (2 joints)
    'neck_pitch':       (-0.7, 0.7),
    'neck_yaw':         (-1.2, 1.2),
}

# MediaPipe PoseLandmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW    = 13
R_ELBOW    = 14
L_WRIST    = 15
R_WRIST    = 16
L_HIP      = 23
R_HIP      = 24
L_KNEE     = 25
R_KNEE     = 26
L_ANKLE    = 27
R_ANKLE    = 28
NOSE       = 0


# ═══════════════════════════════════════════════════════════════
# 1. INITIALIZE MEDIAPIPE POSE LANDMARKER (Tasks API)
# ═══════════════════════════════════════════════════════════════
print("Initializing MediaPipe PoseLandmarker...")

# Store latest detection result (updated asynchronously)
latest_result = [None]
latest_timestamp = [0]

def _pose_callback(result, output_image, timestamp_ms):
    """Callback for async pose detection."""
    latest_result[0] = result
    latest_timestamp[0] = timestamp_ms

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=_pose_callback,
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)
print("  [OK] PoseLandmarker ready")


# ═══════════════════════════════════════════════════════════════
# 2. LOAD THE MUJOCO MODEL
# ═══════════════════════════════════════════════════════════════
print("Loading Angad Humanoid MuJoCo model...")
try:
    model = mujoco.MjModel.from_xml_path("angad_humanoid.xml")
    data = mujoco.MjData(model)
    print(f"  [OK] Model loaded: {model.njnt} joints, {model.nbody} bodies, "
          f"{model.nu} actuators")
except Exception as e:
    print(f"  [ERROR] loading model: {e}")
    print("    Make sure 'angad_humanoid.xml' is in the current directory.")
    exit(1)

# Build actuator ID lookup table
ACT_IDS = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        ACT_IDS[name] = i

print(f"  [OK] {len(ACT_IDS)} actuators indexed")


# ═══════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def manage_joint_limits(target_rad, limit_min, limit_max):
    """
    Clamp target angle to robot joint limits.

    Manages mismatches between human joint ranges and robot URDF limits
    to avoid self-collisions or simulation crashes.
    """
    return float(np.clip(target_rad, limit_min, limit_max))


def smooth_angle(new_val, prev_val, alpha=SMOOTHING_ALPHA):
    """
    Exponential moving average (EMA) filter for smooth joint motion.

    Prevents jerky robot movement from noisy pose estimation.
    Lower alpha = smoother but more lag.
    """
    if prev_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * prev_val


def set_actuator(name, value):
    """
    Set actuator control value by joint name.

    Applies joint limit clamping before setting the control signal.
    The actuator name is derived as 'act_{joint_name}'.
    """
    act_name = f"act_{name}"
    if act_name in ACT_IDS:
        limits = JOINT_LIMITS.get(name)
        if limits:
            value = manage_joint_limits(value, limits[0], limits[1])
        data.ctrl[ACT_IDS[act_name]] = value


def draw_landmarks_on_image(image, pose_result):
    """Draw pose landmarks and connections on the image.

    Custom structural pipelines:
      - Nose: magenta circle for head-tracking reference
      - Left leg (Hip→Knee→Ankle→Heel→Toe): cyan pipeline
      - Right leg (Hip→Knee→Ankle→Heel→Toe): orange pipeline
    """
    if not pose_result or not pose_result.pose_landmarks:
        return image

    L_HEEL, R_HEEL = 29, 30
    L_FOOT_IDX, R_FOOT_IDX = 31, 32

    # Indices for custom highlights (exclude from default drawing)
    SPECIAL_IDS = {NOSE, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, 
                   L_HEEL, R_HEEL, L_FOOT_IDX, R_FOOT_IDX}

    for landmarks in pose_result.pose_landmarks:
        h, w = image.shape[:2]
        # Draw connections
        connections = PoseLandmarksConnections.POSE_LANDMARKS
        for conn in connections:
            start = landmarks[conn.start]
            end = landmarks[conn.end]
            # Don't draw the default thin lines for the legs (we'll draw them thick below)
            is_leg_conn = (conn.start in SPECIAL_IDS) and (conn.end in SPECIAL_IDS) and (conn.start != NOSE)
            
            if start.visibility > 0.3 and end.visibility > 0.3 and not is_leg_conn:
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(image, pt1, pt2, (0, 255, 128), 2)

        # Draw all non-special body landmarks
        for i, lm in enumerate(landmarks):
            if lm.visibility > 0.3:
                cx, cy = int(lm.x * w), int(lm.y * h)
                if i not in SPECIAL_IDS:
                    cv2.circle(image, (cx, cy), 4, (0, 140, 255), -1)
                    cv2.circle(image, (cx, cy), 5, (255, 255, 255), 1)

        # ── Head-tracking highlights (Just nose for visual reference) ──
        if landmarks[NOSE].visibility > 0.05:
            cx, cy = int(landmarks[NOSE].x * w), int(landmarks[NOSE].y * h)
            nose_pt = (cx, cy)
            cv2.circle(image, nose_pt, 8, (255, 0, 255), -1)
            cv2.circle(image, nose_pt, 9, (255, 255, 255), 2)

        # ── Leg-tracking structural pipelines ──
        def _pt(idx):
            lm = landmarks[idx]
            if lm.visibility > 0.3:
                return (int(lm.x * w), int(lm.y * h))
            return None

        lh, lk, la = _pt(L_HIP),   _pt(L_KNEE),   _pt(L_ANKLE)
        lh_eel, lf = _pt(L_HEEL), _pt(L_FOOT_IDX)
        rh, rk, ra = _pt(R_HIP),   _pt(R_KNEE),   _pt(R_ANKLE)
        rh_eel, rf = _pt(R_HEEL), _pt(R_FOOT_IDX)

        # Left Leg (Cyan lines and big dots)
        for pt in [lh, lk, la, lh_eel, lf]:
            if pt:
                cv2.circle(image, pt, 7, (255, 255, 0), -1)
                cv2.circle(image, pt, 9, (255, 255, 255), 2)
        if lh and lk: cv2.line(image, lh, lk, (255, 255, 0), 4)
        if lk and la: cv2.line(image, lk, la, (255, 255, 0), 4)
        if la and lh_eel: cv2.line(image, la, lh_eel, (255, 255, 0), 4)
        if la and lf:     cv2.line(image, la, lf,     (255, 255, 0), 4)
        if lh_eel and lf: cv2.line(image, lh_eel, lf, (255, 255, 0), 4)

        # Right Leg (Orange/Amber lines and big dots)
        for pt in [rh, rk, ra, rh_eel, rf]:
            if pt:
                cv2.circle(image, pt, 7, (0, 165, 255), -1)
                cv2.circle(image, pt, 9, (255, 255, 255), 2)
        if rh and rk: cv2.line(image, rh, rk, (0, 165, 255), 4)
        if rk and ra: cv2.line(image, rk, ra, (0, 165, 255), 4)
        if ra and rh_eel: cv2.line(image, ra, rh_eel, (0, 165, 255), 4)
        if ra and rf:     cv2.line(image, ra, rf,     (0, 165, 255), 4)
        if rh_eel and rf: cv2.line(image, rh_eel, rf, (0, 165, 255), 4)

    return image


def generate_walking_gait(t, ramp=1.0):
    """
    Generate a simple human-like walking gait.
    Returns dict of {joint_name: target_angle_rad}.
    """
    phase = 2.0 * np.pi * WALK_FREQ * t
    r = ramp  # amplitude scale factor

    # ── Forward movement (Y axis is forward for this robot) ──
    root_y = WALK_SPEED * t * ramp

    # ── Hips: SAME sign = alternating (axes are mirrored!) ──
    # Left axis=(-0.94,0,0.34): positive = forward swing
    # Right axis=(+0.94,0,0.34): positive = backward swing
    # SAME value → left forward + right backward = alternating!
    hip_val = WALK_HIP_AMP * r * np.sin(phase)
    hip_l = hip_val
    hip_r = hip_val  # SAME sign — mirrored axes do the alternation

    # ── Knees: bend during swing phase only ──
    # Left swings forward when sin > 0, right when sin < 0
    knee_l = -WALK_KNEE_AMP * r * max(0.0, np.sin(phase))
    knee_r = -WALK_KNEE_AMP * r * max(0.0, -np.sin(phase))

    # ── Ankles: partial compensation ──
    ankle_l = -knee_l * 0.4
    ankle_r = -knee_r * 0.4

    # ── Arms: counter-swing ──
    # Left hip positive = backward swing -> Left arm forward (+ pitch)
    shoulder_l = WALK_ARM_AMP * r * np.sin(phase)
    # Right hip positive = forward swing -> Right arm backward (- pitch)
    shoulder_r = -WALK_ARM_AMP * r * np.sin(phase)
    elbow_l = -WALK_ELBOW_BEND * r
    elbow_r = -WALK_ELBOW_BEND * r

    # ── Torso: slight forward lean ──
    # URDF torso_roll (1 0 0) controls sagittal pitch; negative is forward lean.
    torso_fwd_lean = -WALK_TORSO_LEAN * r

    # ── Head: subtle nod ──
    neck_pitch = 0.03 * r * np.sin(2.0 * phase)

    return {
        'root_x':           0.0,
        'root_y':           root_y,
        'root_yaw':         0.0,
        'hip_pitch_l':      hip_l,
        'hip_pitch_r':      hip_r,
        'knee_pitch_l':     knee_l,
        'knee_pitch_r':     knee_r,
        'ankle_pitch_l':    ankle_l,
        'ankle_pitch_r':    ankle_r,
        'shoulder_pitch_l': shoulder_l,
        'shoulder_pitch_r': shoulder_r,
        'elbow_pitch_l':    elbow_l,
        'elbow_pitch_r':    elbow_r,
        'torso_pitch':      0.0,
        'torso_roll':       torso_fwd_lean,
        'neck_pitch':       neck_pitch,
        'neck_yaw':         0.0,
    }


# ═══════════════════════════════════════════════════════════════
# 4. STATE VARIABLES
# ═══════════════════════════════════════════════════════════════
prev_angles = {}          # Previous frame angles for EMA smoothing

# 3D body capture engine (replaces all manual 2D angle calculations)
bc = BodyCapture(smoothing_alpha=0.35, min_visibility=0.40)

yolo_tr = YoloTracker()
classifier = ActionClassifier()
animator = ActionAnimator()
smpl_tr = SMPLTracker()

# Use a mutable dict so the toggle works correctly inside the loop
state = {
    'mode': 'AI',           # Options: 'MIRROR', 'WALK', 'AI', 'SMPL'
    'start_time': 0.0,
}

# ═══════════════════════════════════════════════════════════════
# 4b. GUI BUTTON HELPERS
# ═══════════════════════════════════════════════════════════════

# Button regions: (x1, y1, x2, y2)
BTN_WALK = {'x1': 10, 'y1': 290, 'x2': 130, 'y2': 330}
BTN_POSE = {'x1': 140, 'y1': 290, 'x2': 290, 'y2': 330}
BTN_AI   = {'x1': 300, 'y1': 290, 'x2': 450, 'y2': 330}
BTN_SMPL = {'x1': 460, 'y1': 290, 'x2': 620, 'y2': 330}


def draw_button(img, btn, text, active=False):
    """Draw a clickable button on the image."""
    if active:
        bg_color = (0, 180, 80)     # green when active
        border_color = (0, 255, 120)
    else:
        bg_color = (60, 60, 65)
        border_color = (120, 120, 130)

    # Filled rectangle (background)
    cv2.rectangle(img, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']),
                  bg_color, -1)
    # Border
    cv2.rectangle(img, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']),
                  border_color, 2)
    # Text centered in button
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = btn['x1'] + (btn['x2'] - btn['x1'] - text_size[0]) // 2
    ty = btn['y1'] + (btn['y2'] - btn['y1'] + text_size[1]) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def point_in_btn(x, y, btn):
    """Check if a click point is inside a button region."""
    return btn['x1'] <= x <= btn['x2'] and btn['y1'] <= y <= btn['y2']


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks on the webcam window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if point_in_btn(x, y, BTN_WALK):
            state['mode'] = 'WALK'
            state['start_time'] = time.time()
            print("  [BTN] WALK clicked")
        elif point_in_btn(x, y, BTN_POSE):
            state['mode'] = 'MIRROR'
            print("  [BTN] POSE clicked")
        elif point_in_btn(x, y, BTN_AI):
            state['mode'] = 'AI'
            print("  [BTN] AI clicked")
        elif point_in_btn(x, y, BTN_SMPL):
            state['mode'] = 'SMPL'
            print("  [BTN] SMPL HD clicked")


# ═══════════════════════════════════════════════════════════════
# 5. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# CAMERA THREAD — captures frames in background at 30 fps
# ═══════════════════════════════════════════════════════════════
class CameraThread:
    """
    Reads webcam frames in a dedicated daemon thread.

    Why: cv2.VideoCapture.read() on Windows blocks until the next
    frame is ready (~33 ms at 30 fps). Calling it directly inside the
    MuJoCo physics loop (which runs at ~500 Hz) starves the camera
    buffer and produces constant 'read failed' errors.

    Solution: thread captures continuously; main loop grabs the cached
    frame instantly with no blocking.
    """
    def __init__(self, index: int = 0):
        # Try DirectShow backend first (most reliable on Windows)
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(index)

        if self._cap.isOpened():
            # Keep buffer at 1 frame — always get the freshest image
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._frame     = None          # latest captured frame (BGR)
        self._lock      = threading.Lock()
        self._running   = self._cap.isOpened()
        self._thread    = threading.Thread(target=self._capture_loop,
                                           daemon=True)
        if self._running:
            self._thread.start()

    def _capture_loop(self):
        """Continuously grab frames; called in background thread."""
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                # Brief pause before retry to avoid busy-spin on error
                time.sleep(0.01)

    def read(self):
        """Return (ok, frame). Thread-safe, non-blocking."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_opened(self):
        return self._running

    def release(self):
        self._running = False
        self._cap.release()


print("Opening webcam (background thread)...")
cam = CameraThread(0)
if cam.is_opened():
    print("  [OK] Camera thread started (DirectShow / 640x480 / 30fps)")
else:
    print("  [WARN] Cannot open webcam. Running simulation without pose input.")
    print("    (Move joints manually in the MuJoCo viewer)")

print("Launching MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("=" * 60)
    print("  OrbitX Task 4 - Dynamic Humanoid Walking Simulation")
    print("  Robot: Angad Full Assembly (23 joints)")
    print("  Pose:  MediaPipe PoseLandmarker (33 landmarks)")
    print("  Hip axis factor: {:.4f}".format(HIP_AXIS_FACTOR))
    print("  Smoothing alpha: {:.2f}".format(SMOOTHING_ALPHA))
    print("  Press 'w' to toggle AUTO-WALK mode")
    print("  Press 'q' in the webcam window to exit")
    print("=" * 60)

    fps_time = time.time()
    frame_count = 0
    fps = 0
    # Use a monotonic counter for MediaPipe timestamps.
    # A wall-clock counter can jitter in thread scheduling and causes
    # MediaPipe LIVE_STREAM mode to silently reject frames.
    video_timestamp_ms = 0

    # Reset walk start time to NOW (not module-load time)
    state['start_time'] = time.time()

    # Register mouse callback for GUI buttons
    cv2.namedWindow('Pilot View - OrbitX Task 4')
    cv2.setMouseCallback('Pilot View - OrbitX Task 4', mouse_callback)

    while viewer.is_running():

        # ═══════════════════════════════════════════════════
        # AUTO-WALK: Apply CPG gait BEFORE camera processing
        # This runs every loop iteration regardless of camera
        # ═══════════════════════════════════════════════════
        if state['mode'] == 'WALK':
            elapsed = time.time() - state['start_time']
            ramp = min(1.0, elapsed / WALK_RAMP_TIME)
            gait = generate_walking_gait(elapsed, ramp)

            for jname, target in gait.items():
                # Root translation/rotation joints must NOT be smoothed
                # (they are linearly increasing positions, not oscillating angles)
                if jname.startswith('root_'):
                    set_actuator(jname, target)
                else:
                    val = smooth_angle(target, prev_angles.get(jname), alpha=0.6)
                    prev_angles[jname] = val
                    set_actuator(jname, val)

            # Debug: print gait values every ~2 seconds
            if int(elapsed * 10) % 20 == 0:
                hip_l = gait.get('hip_pitch_l', 0)
                hip_r = gait.get('hip_pitch_r', 0)
                knee_l = gait.get('knee_pitch_l', 0)
                print(f"  [WALK] t={elapsed:.1f}s ramp={ramp:.2f} "
                      f"hipL={np.degrees(hip_l):+.1f} hipR={np.degrees(hip_r):+.1f} "
                      f"kneeL={np.degrees(knee_l):+.1f}")

        # -- Read and process webcam frame --
        if cam.is_opened():
            ret, frame = cam.read()

            # On read failure: skip pose processing but keep sim alive quietly
            if not ret or frame is None:
                # Create a blank frame so the overlay still renders
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for Camera...", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
                cv2.imshow('Pilot View - OrbitX Task 4', frame)
                cv2.waitKey(1)
                # Still step the physics below — don't break or continue
                result = latest_result[0]
                image = frame.copy()
            else:
                # Monotonic timestamp — MediaPipe LIVE_STREAM requires
                # strictly increasing ms timestamps.
                video_timestamp_ms += 33

                # Convert BGR → RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Send to async landmarker
                pose_landmarker.detect_async(mp_image, video_timestamp_ms)

                # Use the latest result
                result = latest_result[0]
                image = frame.copy()

            # =======================================
            # 1. POSE MIRRORING (MediaPipe IK)
            # =======================================
            if state['mode'] == 'MIRROR' and result is not None:
                if result.pose_landmarks:
                    image = draw_landmarks_on_image(image, result)
                angles_3d = bc.update(result)
                if angles_3d:
                    for jname, val in angles_3d.items():
                        prev_angles[jname] = val
                        set_actuator(jname, val)
                    set_actuator('root_x', 0.0)
                    set_actuator('root_y', 0.0)
                    set_actuator('root_z', 0.0)
                    set_actuator('root_yaw', 3.14159) # 180 degrees turned backwards
            
            # =======================================
            # 2. YOLO AI ACTION MODE
            # =======================================
            elif state['mode'] == 'AI':
                # Run YOLO purely synchronously over the cached frame
                y_res = yolo_tr.process(frame)
                kpts = yolo_tr.get_keypoints(y_res)
                
                # Draw YOLO keypoints visually
                if kpts is not None:
                    for kp in kpts:
                        if kp[2] > 0.5:
                            cx, cy = int(kp[0]), int(kp[1])
                            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                
                # Classify the gesture into an action state
                action = classifier.classify(kpts)
                
                # Print action on screen
                cv2.putText(image, f"AI STATE: {action}", (200, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Run the Action Animator algorithm with dt
                dt_anim = 0.033 # Approx webcam time
                target_angles = animator.step(action, dt_anim)
                
                # ── Live Head Tracking Injection ──
                # Make the head continuously follow the user even during canned animations
                if kpts is not None and kpts[0][2] > 0.4 and kpts[5][2] > 0.4 and kpts[6][2] > 0.4:
                    nose = kpts[0]
                    l_shl = kpts[5]
                    r_shl = kpts[6]
                    mid_shl_x = (l_shl[0] + r_shl[0]) / 2.0
                    mid_shl_y = (l_shl[1] + r_shl[1]) / 2.0
                    shl_width = abs(l_shl[0] - r_shl[0])
                    
                    if shl_width > 10:
                        # Yaw: Nose lateral drift relative to shoulders
                        dx = (nose[0] - mid_shl_x) / shl_width
                        target_angles['neck_yaw'] = np.clip(dx * -1.5, -1.2, 1.2)
                        
                        # Pitch: Up/down tilt (normalize via shoulder width)
                        # Normally nose is ~0.7 to 1.0 shl_width units above shoulders
                        dy = (mid_shl_y - nose[1]) / shl_width
                        target_angles['neck_pitch'] = np.clip((dy - 0.7) * -1.2, -0.7, 0.7)

                for jname, val in target_angles.items():
                    prev_angles[jname] = val
                    set_actuator(jname, val)

                # Lock root to prevent drifting
                set_actuator('root_x', 0.0)
                set_actuator('root_y', 0.0)
                set_actuator('root_z', 0.0)
                set_actuator('root_yaw', 0.0)

            # =======================================
            # 3. SMPL 3D MESH TRACKER
            # =======================================
            elif state['mode'] == 'SMPL':
                if not smpl_tr.active:
                    cv2.putText(image, "SMPL OFFLINE (Missing ROMP pkg)", (100, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    romp_out = smpl_tr.process(frame)
                    if romp_out:
                        smpl_angles = smpl_tr.extract_angles(romp_out)
                        for jname, val in smpl_angles.items():
                            val = smooth_angle(val, prev_angles.get(jname), alpha=0.3)
                            prev_angles[jname] = val
                            set_actuator(jname, val)
                            
                    # Lock root stabilization
                    set_actuator('root_x', 0.0)
                    set_actuator('root_y', 0.0)
                    set_actuator('root_z', 0.0)
                    
                    cv2.putText(image, "SMPL: 3D PRECISION MODE", (180, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

            # =======================================
            # 4. AUTO-WALK (Visual overlay)
            # =======================================
            elif state['mode'] == 'WALK':
                if result and result.pose_landmarks:
                    image = draw_landmarks_on_image(image, result)

            # ── Display info overlay ──
            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = now

            cv2.putText(image, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Mode indicator
            if state['mode'] == 'WALK':
                mode_text = "MODE: AUTO-WALK"
                mode_color = (0, 200, 255)  # orange
            elif state['mode'] == 'MIRROR':
                mode_text = "MODE: POSE MIRROR"
                mode_color = (200, 200, 200)
            elif state['mode'] == 'AI':
                mode_text = "MODE: ACTION AI"
                mode_color = (0, 0, 255)
            elif state['mode'] == 'SMPL':
                mode_text = "MODE: SMPL MESH"
                mode_color = (200, 100, 255)
            
            cv2.putText(image, mode_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
            cv2.putText(image, "OrbitX Task 4 | Angad Humanoid",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Joint angle readout
            y_pos = 105
            display_joints = [
                'shoulder_pitch_l', 'shoulder_pitch_r',
                'elbow_pitch_l', 'elbow_pitch_r',
                'hip_pitch_l', 'hip_pitch_r',
                'knee_pitch_l', 'knee_pitch_r',
                'torso_pitch', 'torso_roll',
                'neck_pitch', 'neck_yaw',
            ]
            for jname in display_joints:
                val = prev_angles.get(jname, 0.0)
                color = (180, 220, 255) if '_l' in jname else (255, 180, 180)
                if 'torso' in jname:
                    color = (200, 200, 200)
                if 'neck' in jname:
                    color = (200, 255, 200)
                cv2.putText(image, f"{jname}: {np.degrees(val):+6.1f} deg",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            color, 1)
                y_pos += 16

            # ── Draw GUI Buttons ──
            draw_button(image, BTN_WALK, "WALK", active=(state['mode'] == 'WALK'))
            draw_button(image, BTN_POSE, "MIRROR", active=(state['mode'] == 'MIRROR'))
            draw_button(image, BTN_AI,   "YOLO AI", active=(state['mode'] == 'AI'))
            draw_button(image, BTN_SMPL, "SMPL HD", active=(state['mode'] == 'SMPL'))

            cv2.imshow('Pilot View - OrbitX Task 4', image)
            # Use 1ms (not 16ms) so we don't block the MuJoCo physics loop.
            # The ~33ms frame rate is already naturally paced by cap.read().
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                state['mode'] = 'WALK' if state['mode'] != 'WALK' else 'MIRROR'
                if state['mode'] == 'WALK':
                    state['start_time'] = time.time()
                    bc.reset()   # clear EMA state when switching modes
                    print("  [KEY] Auto-walk STARTED", flush=True)
                else:
                    bc.reset()   # fresh start for pose mirroring
                    print("  [KEY] Pose mirroring mode (3D BodyCapture)", flush=True)

        # ── Step MuJoCo simulation ──
        # The webcam cap.read() blocks for ~33ms (30 FPS).
        # MuJoCo's dt is 0.002s. We step 16 times (16 * 0.002 = 0.032s)
        # to ensure physics stays running in real-time alongside the camera.
        for _ in range(16):
            mujoco.mj_step(model, data)
        viewer.sync()


# ==============================================================
# CLEANUP
# ==============================================================
cam.release()
cv2.destroyAllWindows()
pose_landmarker.close()
print("\n[OK] Simulation ended cleanly.")
