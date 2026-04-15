import numpy as np

class ActionAnimator:
    """
    Better Simulating Algorithm:
    Procedural trajectory generation based on categorical states.
    Ensures structural stability by interpolating targets smoothly
    instead of jerky direct input mapping.
    """
    def __init__(self):
        self.t = 0.0
        
        # State transitions smoothing
        self.current_angles = {
            'torso_pitch': 0.0, 'torso_roll': 0.0,
            'hip_pitch_l': 0.0, 'hip_pitch_r': 0.0,
            'knee_pitch_l': 0.0, 'knee_pitch_r': 0.0,
            'ankle_pitch_l': 0.0, 'ankle_pitch_r': 0.0,
            'shoulder_pitch_l': 0.0, 'shoulder_pitch_r': 0.0,
            'shoulder_roll_l': 0.0, 'shoulder_roll_r': 0.0,
            'elbow_pitch_l': 0.0, 'elbow_pitch_r': 0.0,
        }

    def _blend(self, target, speed=0.1):
        for k, v in target.items():
            self.current_angles[k] = self.current_angles.get(k, 0.0) * (1 - speed) + v * speed

    def step(self, state: str, dt: float) -> dict:
        self.t += dt
        target = {k: 0.0 for k in self.current_angles.keys()}

        if state == "STANDING":
            # Neutral, relaxed pose
            pass

        elif state == "SQUATTING":
            # Deep knee bend, torso forward to maintain Zero Moment Point (ZMP) center of gravity
            target['hip_pitch_l'] = -0.8
            target['hip_pitch_r'] = -0.8
            target['knee_pitch_l'] = -1.5
            target['knee_pitch_r'] = -1.5
            target['ankle_pitch_l'] = 0.7
            target['ankle_pitch_r'] = 0.7
            target['torso_roll'] = -0.5 # Lean forward
            
            # Arms forward for balance
            target['shoulder_pitch_l'] = -1.0
            target['shoulder_pitch_r'] = -1.0
            target['elbow_pitch_l'] = -0.2
            target['elbow_pitch_r'] = -0.2

        elif state == "WAVING":
            # Right arm raised and oscillating
            target['shoulder_roll_r'] = -1.5  # Lift arm laterally
            # Sine wave oscillation for the wave
            target['shoulder_pitch_r'] = -0.5 + 0.4 * np.sin(self.t * 8.0)
            target['elbow_pitch_r'] = -0.8

        elif state == "ARMS_OUT":
            # Classic T-Pose
            target['shoulder_roll_l'] = 1.5
            target['shoulder_roll_r'] = -1.5

        # Apply procedural interpolation smoothing
        self._blend(target, speed=0.08)
        
        return self.current_angles
