"""
test_walk_only.py — Minimal walking test (NO webcam, NO MediaPipe)
Just the MuJoCo humanoid walking forward.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("angad_humanoid.xml")
data = mujoco.MjData(model)

# Build actuator lookup
ACT = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        ACT[name] = i
print(f"Actuators: {list(ACT.keys())}")


def set_ctrl(joint_name, value):
    """Set actuator control by joint name."""
    act_name = f"act_{joint_name}"
    if act_name in ACT:
        data.ctrl[ACT[act_name]] = value


# ── Walk parameters ──
FREQ = 0.8        # Hz (slow walk)
HIP_AMP = 0.35    # rad (~20 deg)
KNEE_AMP = 0.45   # rad (~26 deg)
ARM_AMP = 0.20    # rad
SPEED = 0.20      # m/s forward

print("\n=== Walking Test ===")
print("The robot should walk forward with alternating legs.")
print("Close the viewer window to stop.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()

    while viewer.is_running():
        t = time.time() - t0
        phase = 2.0 * np.pi * FREQ * t
        ramp = min(1.0, t / 1.5)  # smooth start over 1.5s

        # ── Forward motion ──
        set_ctrl('root_x', SPEED * t * ramp)
        set_ctrl('root_y', 0.0)
        set_ctrl('root_yaw', 0.0)

        # ── HIPS: SAME SIGN = alternating (mirrored axes!) ──
        # Left axis=(-0.94,0,0.34): positive = forward swing
        # Right axis=(+0.94,0,0.34): positive = backward swing
        # So SAME positive value → left forward + right backward
        hip_val = HIP_AMP * ramp * np.sin(phase)
        set_ctrl('hip_pitch_l', hip_val)
        set_ctrl('hip_pitch_r', hip_val)   # SAME sign!

        # ── KNEES: bend during swing phase only ──
        # Left swings forward when sin > 0 → bend left knee then
        # Right swings forward when sin < 0 → bend right knee then
        knee_l = -KNEE_AMP * ramp * max(0.0, np.sin(phase))
        knee_r = -KNEE_AMP * ramp * max(0.0, -np.sin(phase))
        set_ctrl('knee_pitch_l', knee_l)
        set_ctrl('knee_pitch_r', knee_r)

        # ── ANKLES: partial compensation ──
        set_ctrl('ankle_pitch_l', -knee_l * 0.4)
        set_ctrl('ankle_pitch_r', -knee_r * 0.4)

        # ── ARMS: counter-swing ──
        # Left hip positive = backward swing -> Left arm forward (+ pitch)
        set_ctrl('shoulder_pitch_l', ARM_AMP * ramp * np.sin(phase))
        # Right hip positive = forward swing -> Right arm backward (- pitch)
        set_ctrl('shoulder_pitch_r', -ARM_AMP * ramp * np.sin(phase))

        # ── TORSO: slight forward lean ──
        set_ctrl('torso_pitch', 0.0)
        set_ctrl('torso_roll', -0.04 * ramp)

        # Step physics (match real-time)
        for _ in range(16):
            mujoco.mj_step(model, data)
        viewer.sync()

        # Print debug every 2s
        if int(t * 10) % 20 == 0:
            print(f"  t={t:.1f}s  hipL={np.degrees(hip_val):+.1f}  "
                  f"hipR={np.degrees(hip_val):+.1f}  "
                  f"kneeL={np.degrees(knee_l):+.1f}  "
                  f"kneeR={np.degrees(knee_r):+.1f}")

print("\n[OK] Test finished.")
