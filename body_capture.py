# -*- coding: utf-8 -*-
"""
body_capture.py — Full-Body 3D Motion Capture for Angad Humanoid
=================================================================
Accurate joint angle extraction using MediaPipe's 3D world landmarks.

Key improvements over naive 2D approach:
  1. Uses pose_world_landmarks (metric, hip-centred, real-world scale)
     instead of normalised screen (x,y) — removes perspective distortion.
  2. Extracts proper 3D angles via vector cross-products / atan2,
     NOT just 2D arccos between screen points.
  3. Computes all 23 Angad joints:
       Legs   : hip_pitch, thigh_roll, thigh_yaw, knee_pitch,
                ankle_pitch, ankle_roll
       Torso  : torso_pitch, torso_roll, torso_yaw
       Arms   : shoulder_pitch, shoulder_roll, elbow_yaw, elbow_pitch
       Head   : neck_pitch, neck_yaw
  4. Applies per-joint EMA smoothing with configurable alpha.
  5. Hard-clamps every angle to URDF joint limits before returning.

Usage:
    from body_capture import BodyCapture, JOINT_LIMITS
    bc = BodyCapture(smoothing_alpha=0.35, min_visibility=0.4)
    angles = bc.update(pose_result)    # pass MediaPipe result every frame
    # angles is dict {joint_name: float_radians} — use directly with set_ctrl

MediaPipe landmark indices used:
    NOSE=0  L_EAR=7   R_EAR=8
    L_SHOULDER=11  R_SHOULDER=12
    L_ELBOW=13     R_ELBOW=14
    L_WRIST=15     R_WRIST=16
    L_HIP=23       R_HIP=24
    L_KNEE=25      R_KNEE=26
    L_ANKLE=27     R_ANKLE=28
    L_HEEL=29      R_HEEL=30
    L_FOOT_IDX=31  R_FOOT_IDX=32
"""

import numpy as np
from typing import Optional

# ──────────────────────────────────────────────────────────────
# MediaPipe landmark indices
# ──────────────────────────────────────────────────────────────
NOSE        = 0
L_EAR       = 7
R_EAR       = 8
L_SHOULDER  = 11
R_SHOULDER  = 12
L_ELBOW     = 13
R_ELBOW     = 14
L_WRIST     = 15
R_WRIST     = 16
L_HIP       = 23
R_HIP       = 24
L_KNEE      = 25
R_KNEE      = 26
L_ANKLE     = 27
R_ANKLE     = 28
L_HEEL      = 29
R_HEEL      = 30
L_FOOT_IDX  = 31
R_FOOT_IDX  = 32

# ──────────────────────────────────────────────────────────────
# Angad URDF joint limits [min_rad, max_rad]
# ──────────────────────────────────────────────────────────────
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # Legs
    "hip_pitch_l":      (-1.57, 1.57),
    "hip_pitch_r":      (-1.57, 1.57),
    "thigh_roll_l":     (-0.50, 0.50),
    "thigh_roll_r":     (-0.50, 0.50),
    "thigh_yaw_l":      (-0.50, 0.50),
    "thigh_yaw_r":      (-0.50, 0.50),
    "knee_pitch_l":     (-2.00, 2.00),
    "knee_pitch_r":     (-2.00, 2.00),
    "ankle_pitch_l":    (-1.00, 1.00),
    "ankle_pitch_r":    (-1.00, 1.00),
    "ankle_roll_l":     (-0.50, 0.50),
    "ankle_roll_r":     (-0.50, 0.50),
    # Torso
    "torso_pitch":      (-1.57, 1.57),
    "torso_roll":       (-1.57, 1.57),
    "torso_yaw":        (-1.57, 1.57),
    # Arms
    "shoulder_pitch_l": (-3.14, 3.14),
    "shoulder_pitch_r": (-3.14, 3.14),
    "shoulder_roll_l":  ( 0.00, 2.79),
    "shoulder_roll_r":  (-2.88, 0.17),
    "elbow_yaw_l":      (-3.14, 3.14),
    "elbow_yaw_r":      (-3.14, 3.14),
    "elbow_pitch_l":    (-1.57, 1.57),
    "elbow_pitch_r":    (-1.57, 1.57),
    # Head
    "neck_pitch":       (-0.70, 0.70),
    "neck_yaw":         (-1.20, 1.20),
}

# Hip pitch axis tilt factor from URDF
# axis = (-0.939, 0, 0.342) → only 93.9% of angle projects onto sagittal plane
HIP_AXIS_FACTOR = 0.939693


# ──────────────────────────────────────────────────────────────
# Pure math helpers
# ──────────────────────────────────────────────────────────────

def _vec(lm, idx) -> np.ndarray:
    """Extract (x, y, z) from a world landmark list as a numpy vector.

    MediaPipe pose_world_landmarks uses +Y = DOWNWARD.
    We negate Y here so the rest of the code can assume the standard
    robotics / graphics convention of +Y = UPWARD.
    """
    p = lm[idx]
    return np.array([p.x, -p.y, p.z], dtype=np.float64)


def _norm(v: np.ndarray) -> np.ndarray:
    """Normalise a vector; return zero vector on near-zero magnitude."""
    mag = np.linalg.norm(v)
    return v / mag if mag > 1e-8 else np.zeros(3)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle [0, pi] between two direction vectors."""
    cos_a = np.clip(np.dot(_norm(a), _norm(b)), -1.0, 1.0)
    return float(np.arccos(cos_a))


def _signed_angle(v_from: np.ndarray,
                  v_to:   np.ndarray,
                  axis:   np.ndarray) -> float:
    """
    Signed angle from v_from to v_to around 'axis' (right-hand rule).
    Range: (-pi, pi].
    """
    cross = np.cross(v_from, v_to)
    sin_a = np.linalg.norm(cross) * np.sign(np.dot(cross, axis) + 1e-30)
    cos_a = np.dot(_norm(v_from), _norm(v_to))
    return float(np.arctan2(sin_a, cos_a))


def _visible(lm, *indices, threshold: float = 0.4) -> bool:
    """Return True only if every landmark meets the visibility threshold."""
    return all(lm[i].visibility >= threshold for i in indices)


# ──────────────────────────────────────────────────────────────
# BodyCapture — main class
# ──────────────────────────────────────────────────────────────

class BodyCapture:
    """
    Full-body 3D joint angle extractor for the Angad humanoid.

    Parameters
    ----------
    smoothing_alpha : float
        EMA coefficient (0 = frozen, 1 = raw). 0.35 is a good default.
    min_visibility : float
        Minimum MediaPipe landmark visibility score to trust a point.
    """

    def __init__(self, smoothing_alpha: float = 0.35,
                 min_visibility: float = 0.40):
        self.alpha   = smoothing_alpha
        self.min_vis = min_visibility
        self._prev: dict[str, float] = {}   # EMA state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, pose_result) -> dict[str, float]:
        """
        Extract joint angles from a MediaPipe PoseLandmarker result.

        Parameters
        ----------
        pose_result : mediapipe PoseLandmarkerResult (or None)

        Returns
        -------
        dict mapping joint_name -> angle_in_radians (clamped to URDF limits)
        """
        if pose_result is None:
            return {}
        if not pose_result.pose_world_landmarks:
            return {}

        # Use the first detected person's world landmarks (metric, hip-centred)
        lm = pose_result.pose_world_landmarks[0]
        angles: dict[str, float] = {}

        # ── COORDINATE SYSTEM NOTE ────────────────────────────────────
        # MediaPipe world coords (pose_world_landmarks):
        #   +X = rightward (from subject's perspective)
        #   +Y = upward
        #   +Z = toward camera
        # Origin = midpoint of hips
        # Units = metres (approximate)
        # ─────────────────────────────────────────────────────────────

        self._extract_torso(lm, angles)
        self._extract_left_leg(lm, angles)
        self._extract_right_leg(lm, angles)
        self._extract_left_arm(lm, angles)
        self._extract_right_arm(lm, angles)
        self._extract_head(lm, angles)

        # Apply EMA smoothing + clamp to joint limits
        result: dict[str, float] = {}
        for name, raw in angles.items():
            smoothed = self._adaptive_smooth(name, raw)
            lo, hi   = JOINT_LIMITS.get(name, (-3.14, 3.14))
            result[name] = float(np.clip(smoothed, lo, hi))

        return result

    # ------------------------------------------------------------------
    # Torso
    # ------------------------------------------------------------------

    def _extract_torso(self, lm, out: dict):
        """
        Compute torso_pitch (sagittal lean), torso_roll (lateral lean),
        torso_yaw (axial twist) from shoulder/hip plane geometry.
        """
        if not _visible(lm, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER,
                         threshold=self.min_vis):
            return

        l_hip  = _vec(lm, L_HIP)
        r_hip  = _vec(lm, R_HIP)
        l_shl  = _vec(lm, L_SHOULDER)
        r_shl  = _vec(lm, R_SHOULDER)

        mid_hip = (l_hip  + r_hip)  / 2.0
        mid_shl = (l_shl  + r_shl)  / 2.0

        # Spine vector (hip mid → shoulder mid)
        spine   = _norm(mid_shl - mid_hip)

        # Reference axes
        up      = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])   # toward camera
        right   = np.array([1.0, 0.0, 0.0])

        # Pitch = forward/backward lean
        computed_pitch = float(np.arctan2(spine[2], spine[1]))

        # Roll = lateral lean
        computed_roll  = float(np.arctan2(spine[0], spine[1]))

        # The robot's URDF names use aircraft-style labelling:
        # axis "1 0 0" (Lateral X axis) is named "torso_roll", but physically produces Pitch (forward/back)
        # axis "0 1 0" (Forward Y axis) is named "torso_pitch", but physically produces Roll (sideways)
        # We swap the assignments here to map anatomical motion to the physical axis.
        out["torso_pitch"] = computed_roll    # Drives the 0 1 0 joint (sideways Roll)
        out["torso_roll"]  = computed_pitch   # Drives the 1 0 0 joint (forward Pitch)

        # Yaw = twist: compare shoulder line direction to hip line direction
        hip_axis  = _norm(r_hip - l_hip)
        shl_axis  = _norm(r_shl - l_shl)
        out["torso_yaw"]   = _signed_angle(hip_axis, shl_axis, up)

    # ------------------------------------------------------------------
    # Legs (shared helper)
    # ------------------------------------------------------------------

    def _extract_leg(self, lm, side: str, out: dict):
        """
        Compute all 6 leg joints for one side.
        side = 'l' or 'r'
        """
        s = side
        HIP    = L_HIP    if s == 'l' else R_HIP
        KNEE   = L_KNEE   if s == 'l' else R_KNEE
        ANKLE  = L_ANKLE  if s == 'l' else R_ANKLE
        HEEL   = L_HEEL   if s == 'l' else R_HEEL
        FOOT   = L_FOOT_IDX if s == 'l' else R_FOOT_IDX

        if not _visible(lm, HIP, KNEE, ANKLE, threshold=0.6):
            # Strict fallback: if legs aren't clearly visible, force them to neutral 0.0
            # This prevents the robot's knees from breaking when occluded
            out[f"hip_pitch_{s}"]   = 0.0
            out[f"thigh_roll_{s}"]  = 0.0
            out[f"thigh_yaw_{s}"]   = 0.0
            out[f"knee_pitch_{s}"]  = 0.0
            out[f"ankle_pitch_{s}"] = 0.0
            out[f"ankle_roll_{s}"]  = 0.0
            return

        hip_p   = _vec(lm, HIP)
        knee_p  = _vec(lm, KNEE)
        ankle_p = _vec(lm, ANKLE)

        thigh_v = _norm(knee_p  - hip_p)    # points down the thigh
        shin_v  = _norm(ankle_p - knee_p)   # points down the shin

        # ── Hip pitch (sagittal swing, forward/backward) ──────────────
        # Angle between vertical (down) and thigh in sagittal plane
        down    = np.array([0.0, -1.0, 0.0])
        fwd     = np.array([0.0,  0.0,  1.0])
        # Signed around mediolateral axis (X)
        lat     = np.array([1.0,  0.0,  0.0]) * (-1 if s == 'l' else 1)
        hip_pitch = _signed_angle(down, thigh_v, lat)
        # Apply hip axis factor (axis is tilted in URDF)
        # Angad axis convention: same sign = alternating (see xml comments)
        out[f"hip_pitch_{s}"] = hip_pitch / HIP_AXIS_FACTOR

        # ── Thigh roll (frontal abduction/adduction) ───────────────────
        # Project thigh onto frontal plane (Y-Z), measure X deviation
        thigh_roll = float(np.arctan2(thigh_v[0], -thigh_v[1]))
        out[f"thigh_roll_{s}"] = thigh_roll * (-1 if s == 'r' else 1)

        # ── Thigh yaw (axial / internal-external rotation) ────────────
        # MediaPipe's depth estimation for legs is highly unstable when facing 
        # the camera, causing the hip to randomly twist 180 degrees inward or 
        # outward. We lock the hip yaw to 0.0 to guarantee structural stability 
        # (knees always bend straight backward as expected).
        out[f"thigh_yaw_{s}"] = 0.0

        # ── Knee pitch (flexion — always negative = bend) ──────────────
        # Full 3D angle between thigh and shin vectors
        knee_angle = _angle_between(-thigh_v, shin_v)   # 0=straight, pi=fully bent
        out[f"knee_pitch_{s}"] = -(np.pi - knee_angle)  # negative = flexion

        # ── Ankle pitch (dorsiflexion / plantarflexion) ───────────────
        if _visible(lm, ANKLE, HEEL, FOOT, threshold=self.min_vis):
            heel_p  = _vec(lm, HEEL)
            foot_p  = _vec(lm, FOOT)
            foot_v  = _norm(foot_p - heel_p)   # toe direction
            # Angle between shin and foot in sagittal plane
            ankle_pitch = _signed_angle(-shin_v, foot_v, lat)
            out[f"ankle_pitch_{s}"] = ankle_pitch * 0.6  # scale to robot range
        else:
            # Fallback: partial compensation from knee + hip
            knee_val = out.get(f"knee_pitch_{s}", 0.0)
            hp_val   = out.get(f"hip_pitch_{s}",  0.0) * HIP_AXIS_FACTOR
            out[f"ankle_pitch_{s}"] = -(hp_val + knee_val) * 0.35

        # ── Ankle roll (inversion / eversion) ─────────────────────────
        if _visible(lm, HEEL, FOOT, threshold=self.min_vis):
            heel_p = _vec(lm, HEEL)
            foot_p = _vec(lm, FOOT)
            foot_v = _norm(foot_p - heel_p)
            ankle_roll = float(np.arctan2(foot_v[0], foot_v[2])) * 0.4
            out[f"ankle_roll_{s}"] = ankle_roll * (-1 if s == 'r' else 1)
        else:
            out[f"ankle_roll_{s}"] = 0.0

    def _extract_left_leg(self, lm, out):
        self._extract_leg(lm, 'l', out)

    def _extract_right_leg(self, lm, out):
        self._extract_leg(lm, 'r', out)

    # ------------------------------------------------------------------
    # Arms (shared helper)
    # ------------------------------------------------------------------

    def _extract_arm(self, lm, side: str, out: dict):
        """
        Compute 4 arm joints for one side.
        """
        s = side
        SHOULDER = L_SHOULDER if s == 'l' else R_SHOULDER
        ELBOW    = L_ELBOW    if s == 'l' else R_ELBOW
        WRIST    = L_WRIST    if s == 'l' else R_WRIST
        
        if not _visible(lm, SHOULDER, ELBOW, threshold=self.min_vis):
            return

        shl_p   = _vec(lm, SHOULDER)
        elb_p   = _vec(lm, ELBOW)
        upper_v = _norm(elb_p - shl_p)   # shoulder → elbow

        # Define robust global body axes
        l_shl = _vec(lm, L_SHOULDER)
        r_shl = _vec(lm, R_SHOULDER)
        up    = np.array([0.0, 1.0, 0.0])
        # body_left points to the user's left (+X)
        body_left = _norm(l_shl - r_shl)
        # body_fwd points towards the camera (-Z)
        body_fwd  = _norm(np.cross(up, body_left))
        
        # Outward lateral axis for this specific arm
        lat_outward = body_left if s == 'l' else -body_left

        # ── Shoulder pitch (forward/backward swing) ────────────────────
        # Angle of upper arm around the lateral axis
        down      = np.array([0.0, -1.0, 0.0])
        # _signed_angle(down, upper_v, lat_outward) -> right hand rule
        # For left arm (lat points +X): moving arm forward (-Z) means right-hand rule points -Y.
        # Wait, if moving forward into -Z, down x (-Z) = (0, -1, 0) x (0, 0, -1) = (1, 0, 0) = +X!
        # So sin_a is positive. Pitch is positive. Let's just use it consistently.
        shl_pitch = _signed_angle(down, upper_v, lat_outward)
        out[f"shoulder_pitch_{s}"] = shl_pitch

        # ── Shoulder roll (abduction / adduction) ──────────────────────
        # Project upper arm onto the plane formed by `up` and `lat_outward`
        shl_roll = float(np.arctan2(
            np.dot(upper_v, lat_outward),  # outward component
            -upper_v[1]                    # downward component
        ))
        out[f"shoulder_roll_{s}"] = shl_roll if s == 'l' else -shl_roll

        # ── Elbow pitch (flexion) ──────────────────────────────────────
        if _visible(lm, ELBOW, WRIST, threshold=self.min_vis):
            wrist_p   = _vec(lm, WRIST)
            forearm_v = _norm(wrist_p - elb_p)
            elbow_ang = _angle_between(-upper_v, forearm_v)
            out[f"elbow_pitch_{s}"] = -(np.pi - elbow_ang)
        else:
            out[f"elbow_pitch_{s}"] = -0.3

        # ── Elbow yaw (forearm rotation) ───────────────────────────────
        if _visible(lm, ELBOW, WRIST, threshold=self.min_vis):
            wrist_p   = _vec(lm, WRIST)
            forearm_v = _norm(wrist_p - elb_p)
            
            # Singularity-free reference: use body_left instead of arm-specific lat
            # so the "up" reference vector never flips upside down between arms.
            ref_up   = _norm(np.cross(body_left, upper_v))
            ref_fwd  = _norm(np.cross(upper_v, ref_up))
            
            proj_y = np.dot(forearm_v, ref_up)
            proj_x = np.dot(forearm_v, ref_fwd)
            
            elbow_yaw = float(np.arctan2(proj_x, proj_y))
            out[f"elbow_yaw_{s}"] = float(np.clip(elbow_yaw, -1.5, 1.5))
        else:
            out[f"elbow_yaw_{s}"] = 0.0

    def _extract_left_arm(self, lm, out):
        self._extract_arm(lm, 'l', out)

    def _extract_right_arm(self, lm, out):
        self._extract_arm(lm, 'r', out)

    # ------------------------------------------------------------------
    # Head
    # ------------------------------------------------------------------

    def _extract_head(self, lm, out: dict):
        """
        Compute neck_pitch (nod) and neck_yaw (turn).

        Face landmarks in pose_world_landmarks often have very low visibility
        scores, so we use much lower thresholds than the body joints.
        """
        HEAD_VIS = 0.10    # Low threshold for nose

        # ── Neck pitch (nod: positive = looking down) ──────────────────
        # Rely exclusively on Nose to prevent Ear-tracking glitches.
        if _visible(lm, NOSE, threshold=HEAD_VIS) and \
           _visible(lm, L_SHOULDER, R_SHOULDER, threshold=self.min_vis):
            mid_shl = (_vec(lm, L_SHOULDER) + _vec(lm, R_SHOULDER)) / 2.0
            nose_p  = _vec(lm, NOSE)
            head_v  = _norm(nose_p - mid_shl)
            up      = np.array([0.0, 1.0, 0.0])
            body_left = _norm(_vec(lm, L_SHOULDER) - _vec(lm, R_SHOULDER))
            raw_pitch = _signed_angle(up, head_v, body_left)
            # Subtract neutral offset (adjusted to keep head straight by default)
            neck_pitch = (raw_pitch - 0.45) * 0.8
            out["neck_pitch"] = float(np.clip(neck_pitch, -0.7, 0.7))

        # ── Neck yaw (turn: positive = looking right) ──────────────────
        # Estimate yaw purely from the NOSE's lateral offset relative to the shoulders
        if _visible(lm, NOSE, threshold=HEAD_VIS) and \
             _visible(lm, L_SHOULDER, R_SHOULDER, threshold=self.min_vis):
            mid_shl = (_vec(lm, L_SHOULDER) + _vec(lm, R_SHOULDER)) / 2.0
            nose_p  = _vec(lm, NOSE)
            shl_width = np.linalg.norm(
                _vec(lm, R_SHOULDER) - _vec(lm, L_SHOULDER))
            if shl_width > 0.01:
                # If nose is left of center, dx is positive (User's Left = +X)
                dx = nose_p[0] - mid_shl[0]
                # Positive dx (left) -> Negative Yaw?
                # Actually, our yaw convention: Positive = Right. Left = Negative. So -dx.
                head_yaw = float(np.clip(-dx / (shl_width * 0.5) * 1.5, -1.2, 1.2))
                out["neck_yaw"] = head_yaw

    # ------------------------------------------------------------------
    # Adaptive Kinematic Smoothing
    # ------------------------------------------------------------------

    def _adaptive_smooth(self, name: str, raw: float) -> float:
        """
        Velocity-based adaptive Exponential Moving Average.
        - High velocity = lower alpha (faster response, no lag)
        - Low velocity  = higher alpha (strong smoothing, no jitter)
        """
        prev = self._prev.get(name)
        if prev is None:
            self._prev[name] = raw
            return raw

        # Calculate joint jump (delta per frame)
        delta = abs(raw - prev)
        
        # 1. Glitch Rejection: If joint jumps more than ~45 degrees in 1 frame (33ms)
        # It is physically impossible for a human. MediaPipe has glitched. Ignore it completely.
        if delta > 0.8:
            return prev

        # 2. Deadband: If movement is tiny (< 1 degree), it's just pixel noise.
        # Lock the joint completely to stop vibration.
        if delta < 0.015:
            return prev
            
        # 3. Dynamic Smoothing (User requested slower, smoother movement)
        if delta < 0.10:
            # Slow movement / moderate noise -> Extremely Heavy smoothing (gliding)
            current_alpha = 0.98
        elif delta > 0.30:
            # Fast movement -> Still heavy smoothing (no snapping, just deliberate swinging)
            current_alpha = 0.85
        else:
            # Interpolate smoothly
            progress = (delta - 0.10) / 0.20
            current_alpha = 0.98 - (progress * 0.13)

        # Apply smoothing
        smoothed = current_alpha * prev + (1.0 - current_alpha) * raw
        self._prev[name] = smoothed
        return smoothed

    def reset(self):
        """Reset state (call when robot teleports or mode switches)."""
        self._prev.clear()
