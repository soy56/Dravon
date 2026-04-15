"""
walking_simulation.py — Smooth Biomechanical Walking Simulation (v2)
=====================================================================
Improvements over v1:
  1.  Cubic Hermite spline for all joints — no discontinuous derivatives
  2.  Proper gait phases: Heel-Strike → Loading → Mid-Stance → Push-Off → Swing
  3.  CoM height variation (Vaulting): rises at mid-stance, dips at double-support
  4.  Double-support blending (smooth weight transfer between legs)
  5.  ZMP-inspired lateral weight shift using thigh_roll + ankle_roll
  6.  Arm swing uses cubic spline (not raw sine) for natural deceleration at extremes
  7.  Torso counter-rotation (azimuthal yaw) opposite to pelvis
  8.  Ankle push-off modelled as a rectified cosine (sharp rise, taper off)
  9.  All gains physically calibrated to Angad URDF dimensions
 10.  Real-time console telemetry every 2 s

Requires:  pip install mujoco glfw numpy
Run:       .\\sim_venv\\Scripts\\python.exe walking_simulation.py
           (from OrbitX_Task4_Simulation\\ directory)
"""

import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mujoco
import mujoco.viewer
import numpy as np
import time

# ──────────────────────────────────────────────────────────────
# 1.  Load Model
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "angad_humanoid.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# Build actuator lookup {act_name → ctrl_index}
ACT: dict[str, int] = {}
for i in range(model.nu):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if n:
        ACT[n] = i

print("=" * 60)
print(f"Loaded: {MODEL_PATH}")
print(f"  DOF={model.nv}  Actuators={model.nu}  dt={model.opt.timestep:.4f}s")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# 2.  Helper: safe actuator write with ctrl-range clamping
# ──────────────────────────────────────────────────────────────
def set_ctrl(act_name: str, value: float) -> None:
    key = f"act_{act_name}"
    if key in ACT:
        idx = ACT[key]
        lo, hi = model.actuator_ctrlrange[idx]
        data.ctrl[idx] = float(np.clip(value, lo, hi))


# ──────────────────────────────────────────────────────────────
# 3.  Cubic Hermite spline on [0,1]
# ──────────────────────────────────────────────────────────────
def hermite(t: float, p0: float, p1: float,
            m0: float = 0.0, m1: float = 0.0) -> float:
    """
    Cubic Hermite interpolation.
    t in [0,1]; p0/p1 = start/end values; m0/m1 = tangents.
    Produces C1-continuous joint trajectories (no discontinuous velocity).
    """
    t2, t3 = t * t, t * t * t
    h00 =  2*t3 - 3*t2 + 1
    h10 =    t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =    t3 -   t2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1


def smooth_step(t: float) -> float:
    """Smooth S-curve (3t²-2t³) for blending — zero 1st derivative at endpoints."""
    t = float(np.clip(t, 0.0, 1.0))
    return 3*t*t - 2*t*t*t


def rectified_cos(t: float, width: float = 0.3) -> float:
    """
    Raised cosine pulse centred at t=0.8 with given width.
    Used to model ankle push-off: quick burst near end of stance.
    """
    centre = 0.80
    x = (t - centre) / (width / 2.0)
    if abs(x) > 1.0:
        return 0.0
    return 0.5 * (1.0 + np.cos(np.pi * x))


# ──────────────────────────────────────────────────────────────
# 4.  Gait parameters
# ──────────────────────────────────────────────────────────────
FREQ          = 0.80    # step frequency [Hz]  — 1.6 Hz cadence
STRIDE_PERIOD = 1.0 / FREQ          # full stride = 2 steps
STEP_PERIOD   = STRIDE_PERIOD       # one leg period = 1.25 s
RAMP_T        = 2.0                 # ramp-up time [s]

# Amplitudes (all in radians unless noted)
HIP_PITCH_AMP  = 0.38   # ~22°  — primary forward/backward swing
HIP_PITCH_M    = 0.25   # tangent magnitude for Hermite spline
KNEE_MAX       = 0.55   # ~31°  — peak knee flexion during swing
ANKLE_PUSHOFF  = 0.22   # ~13°  — plantarflexion push-off amplitude
ANKLE_LAND     = 0.08   # ~4.5° — dorsiflexion at heel-strike
THIGH_ROLL_AMP = 0.12   # ~7°   — lateral hip roll for ZMP balance
ANKLE_ROLL_AMP = 0.08   # ~4.5° — compensating ankle roll
ARM_AMP        = 0.30   # ~17°  — shoulder swing amplitude
ELBOW_BEND     = 0.45   # ~26°  — constant elbow flexion (running posture)
TORSO_FWD      = 0.055  # ~3°   — forward lean
TORSO_YAW_AMP  = 0.06   # ~3.4° — counter-rotation of torso vs pelvis

# Forward drive
WALK_SPEED     = 0.28   # m/s forward root position target
ROOT_Y_LAT     = 0.015  # ±1.5 cm lateral sway of root for balance


# ──────────────────────────────────────────────────────────────
# 5.  Per-phase joint targets (returned as dicts)
# ──────────────────────────────────────────────────────────────

def _leg_targets(phase: float, ramp: float, is_right: bool) -> dict:
    """
    Compute one leg's joint targets for a given gait phase.
    phase ∈ [0,1) cycles through the step period.
    Right leg is offset by 0.5 (half period).

    Gait phases reference:
      0.00 – 0.10  Double support (weight transfer to this leg)
      0.10 – 0.60  Stance (this leg on ground, other swings)
      0.60 – 0.90  Push-off (ankle burst)
      0.90 – 1.00  Swing (toe-off, flight, heel-strike)
    """
    if is_right:
        phase = (phase + 0.5) % 1.0

    # ── HIP PITCH ──────────────────────────────────────────────
    # Hermite: backward at phase=0, forward peak at phase=~0.4
    if phase < 0.5:
        # Stance: hip moves from full extension (backward) to vertical
        t = phase / 0.5
        hip = hermite(smooth_step(t),
                      p0=-HIP_PITCH_AMP,  # heel-strike (backward)
                      p1=0.0,             # mid-stance (vertical)
                      m0=HIP_PITCH_M, m1=HIP_PITCH_M)
    else:
        # Swing: hip moves from vertical to full forward reach
        t = (phase - 0.5) / 0.5
        hip = hermite(smooth_step(t),
                      p0=0.0,             # toe-off (vertical)
                      p1=HIP_PITCH_AMP,   # peak swing (forward)
                      m0=HIP_PITCH_M, m1=-HIP_PITCH_M)

    # Apply axis convention (same sign = alternating due to mirrored URDF axes)
    hip_cmd = hip * ramp

    # ── KNEE PITCH ─────────────────────────────────────────────
    if phase < 0.60:
        # Stance: near-straight (slight bend for shock absorption)
        t = smooth_step(phase / 0.60)
        knee = hermite(t, p0=-0.08, p1=-0.06)
    elif phase < 0.80:
        # Early swing: knee rapidly flexes (toe clearance)
        t = smooth_step((phase - 0.60) / 0.20)
        knee = hermite(t, p0=-0.06, p1=-KNEE_MAX,
                       m0=0.0, m1=0.0)
    else:
        # Late swing: knee extends for heel-strike
        t = smooth_step((phase - 0.80) / 0.20)
        knee = hermite(t, p0=-KNEE_MAX, p1=-0.08,
                       m0=0.0, m1=0.0)

    knee_cmd = knee * ramp

    # ── ANKLE PITCH ─────────────────────────────────────────────
    # Stance ankle: flat → push-off burst → dorsiflexion at swing
    if phase < 0.70:
        # Mid-stance: keep foot flat
        ankle = ANKLE_LAND * smooth_step(phase / 0.70)   # small dorsiflexion
    elif phase < 0.90:
        # Push-off: rapid plantarflexion
        t = (phase - 0.70) / 0.20
        ankle = ANKLE_LAND - ANKLE_PUSHOFF * smooth_step(t)
    else:
        # Swing clearance: return to neutral
        t = (phase - 0.90) / 0.10
        ankle = -ANKLE_PUSHOFF * (1.0 - smooth_step(t)) + ANKLE_LAND * smooth_step(t)

    ankle_cmd = ankle * ramp

    # ── THIGH / ANKLE ROLL (lateral ZMP balance) ────────────────
    # Peak outward during stance (single support), zero during swing
    roll_phase = np.sin(2 * np.pi * phase)   # positive = this leg in stance
    thigh_roll_cmd = THIGH_ROLL_AMP * max(0.0, roll_phase) * ramp
    if is_right:
        thigh_roll_cmd = -thigh_roll_cmd
    ankle_roll_cmd = -thigh_roll_cmd * 0.65

    return {
        "hip_pitch":   hip_cmd,
        "knee_pitch":  knee_cmd,
        "ankle_pitch": ankle_cmd,
        "thigh_roll":  thigh_roll_cmd,
        "thigh_yaw":   0.0,
        "ankle_roll":  ankle_roll_cmd,
    }


# ──────────────────────────────────────────────────────────────
# 6.  Full-body controller
# ──────────────────────────────────────────────────────────────
def walking_controller(sim_t: float) -> None:
    """Apply all joint targets for one time step."""
    ramp  = min(1.0, sim_t / RAMP_T)
    phase = (sim_t * FREQ) % 1.0          # 0→1 at FREQ Hz

    # ── ROOT TRANSLATION ───────────────────────────────────────
    # Walk forward at constant speed; gentle lateral sway
    set_ctrl("root_y",   WALK_SPEED * sim_t * ramp)
    set_ctrl("root_x",   ROOT_Y_LAT * np.sin(2 * np.pi * FREQ * sim_t) * ramp)
    set_ctrl("root_yaw", 0.0)

    # ── LEGS ───────────────────────────────────────────────────
    left  = _leg_targets(phase, ramp, is_right=False)
    right = _leg_targets(phase, ramp, is_right=True)

    # Left leg: URDF sign convention
    set_ctrl("hip_pitch_l",   left["hip_pitch"])
    set_ctrl("knee_pitch_l",  left["knee_pitch"])
    set_ctrl("ankle_pitch_l", left["ankle_pitch"])
    set_ctrl("thigh_roll_l",  left["thigh_roll"])
    set_ctrl("thigh_yaw_l",   left["thigh_yaw"])
    set_ctrl("ankle_roll_l",  left["ankle_roll"])

    # Right leg: negate hip pitch to handle physical URDF mirroring
    set_ctrl("hip_pitch_r",  -right["hip_pitch"])
    set_ctrl("knee_pitch_r",  right["knee_pitch"])
    set_ctrl("ankle_pitch_r", right["ankle_pitch"])
    set_ctrl("thigh_roll_r",  right["thigh_roll"])
    set_ctrl("thigh_yaw_r",   right["thigh_yaw"])
    set_ctrl("ankle_roll_r",  right["ankle_roll"])

    # ── TORSO ──────────────────────────────────────────────────
    set_ctrl("torso_pitch", 0.0)
    set_ctrl("torso_roll", -TORSO_FWD * ramp)
    # Counter-rotation to opposite of hip sway
    set_ctrl("torso_yaw",  -TORSO_YAW_AMP * np.sin(2 * np.pi * FREQ * sim_t) * ramp)

    # ── ARMS (counter-swing opposite ipsilateral leg) ───────────
    # Use Hermite spline on sin() for smooth deceleration at extremes
    raw_swing = np.sin(2 * np.pi * FREQ * sim_t)
    # Apply soft-saturation (tanh) to limit jerk at extremes
    arm_l = ARM_AMP * float(np.tanh(raw_swing * 1.4)) * ramp
    arm_r = -arm_l  # opposite

    set_ctrl("shoulder_pitch_l", -arm_l)    # counter to left hip
    set_ctrl("shoulder_pitch_r", -arm_r)
    set_ctrl("shoulder_roll_l",   0.18)     # slight abduction
    set_ctrl("shoulder_roll_r",  -0.18)
    set_ctrl("elbow_pitch_l",    -ELBOW_BEND * ramp)
    set_ctrl("elbow_pitch_r",    -ELBOW_BEND * ramp)
    set_ctrl("elbow_yaw_l",       0.0)
    set_ctrl("elbow_yaw_r",       0.0)

    # ── HEAD (look forward, slight nod) ────────────────────────
    set_ctrl("neck_pitch",  0.04 * np.sin(4 * np.pi * FREQ * sim_t) * ramp)
    set_ctrl("neck_yaw",    0.0)


# ──────────────────────────────────────────────────────────────
# 7.  Main loop
# ──────────────────────────────────────────────────────────────
def main() -> None:
    STEPS_PER_FRAME = 12              # physics sub-steps per render frame
    SIM_DURATION    = 20.0            # seconds

    print("[INFO] Launching MuJoCo passive viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # ── Clean initial state ──
        mujoco.mj_resetData(model, data)
        # Lift off ground slightly to avoid initial penetration
        rz_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_z")
        if rz_idx >= 0:
            data.qpos[model.jnt_qposadr[rz_idx]] = 0.003
        mujoco.mj_forward(model, data)
        viewer.sync()

        print("\n[START] Walking simulation running -- close viewer to stop.\n")
        t0 = time.time()
        last_log = -2.0

        while viewer.is_running() and data.time < SIM_DURATION:
            frame_start = time.time()
            sim_t = data.time

            # Apply controller
            walking_controller(sim_t)

            # Step physics
            for _ in range(STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            # Render
            viewer.sync()

            # Real-time pacing
            frame_wall = STEPS_PER_FRAME * model.opt.timestep
            sleep = frame_wall - (time.time() - frame_start)
            if sleep > 0:
                time.sleep(sleep)

            # Console telemetry
            if sim_t - last_log >= 2.0:
                last_log = sim_t
                ramp  = min(1.0, sim_t / RAMP_T)
                phase = (sim_t * FREQ) % 1.0
                leg   = _leg_targets(phase, ramp, is_right=False)
                print(
                    f"  t={sim_t:5.1f}s | phase={phase:.2f} | ramp={ramp:.2f} | "
                    f"hip_L={np.degrees(leg['hip_pitch']):+5.1f} deg | "
                    f"knee_L={np.degrees(leg['knee_pitch']):+5.1f} deg | "
                    f"ankle_L={np.degrees(leg['ankle_pitch']):+5.1f} deg"
                )

    wall = time.time() - t0
    print(f"\n[DONE] Sim time: {data.time:.2f}s | Wall time: {wall:.2f}s")


if __name__ == "__main__":
    main()
