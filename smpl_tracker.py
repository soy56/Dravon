import numpy as np
import time

try:
    import romp
    ROMP_AVAILABLE = True
except ImportError:
    ROMP_AVAILABLE = False
    print("  [WARN] SMPL 'simple-romp' is not installed. SMPL HD mode will be disabled.")


class SMPLTracker:
    def __init__(self):
        """
        SMPL 3D Mesh Reconstruction Wrapper.
        Extracts 72 Euler/Quaternion joint parameters directly from the image
        by fitting a statistical 3D topological body mesh (ROMP).
        """
        self.active = ROMP_AVAILABLE
        self.model = None

        if self.active:
            print("Loading ROMP SMPL Mesh Estimator...")
            # Initialize with default settings (auto-detects GPU)
            try:
                settings = romp.main.default_settings()
            except TypeError:
                # ROMP v1.x fallback
                settings = romp.main.default_settings
            
            # Disable visualization rendering to save massively on FPS
            settings.show = False
            settings.save_video = False
            settings.render_mesh = False
            
            try:
                self.model = romp.ROMP(settings)
                print("  [OK] ROMP SMPL Initialized")
            except Exception as e:
                print(f"  [ERROR] ROMP failed to initialize: {e}")
                print("  Please download required SMPL_NEUTRAL.pth files into ~/.romp/")
                self.active = False
                self.model = None

    def process(self, frame_bgr):
        """
        Returns a dictionary of SMPL parameters from ROMP.
        Key outputs typically include:
          - 'smpl_thetas': Shape (1, 72) -> 24 joints * 3 axis rotations (axis-angle)
          - 'cam': Shape (1, 3) -> camera translation estimates
        """
        if not self.active or self.model is None:
            return None
            
        try:
            # Output is a dict with keys like 'smpl_thetas', 'smpl_beta', 'cam', etc.
            outputs = self.model(frame_bgr)
            if outputs is None or len(outputs) == 0:
                return None
            return outputs
        except Exception as e:
            print(f"  [ROMP Error] {e}")
            return None

    def extract_angles(self, romp_outputs) -> dict:
        """
        Maps the 72 SMPL axis-angle parameters to the Angad Humanoid 23 DOF URDF structure.
        """
        if not romp_outputs or 'smpl_thetas' not in romp_outputs:
            return {}

        # smpl_thetas is typically (Num_people, 72). We take person 0.
        thetas = romp_outputs['smpl_thetas'][0]
        
        # In SMPL:
        # 0: Pelvis (Root)
        # 1: L_Hip, 2: R_Hip
        # 3: Spine1, # 6: Spine2, # 9: Spine3
        # 4: L_Knee, 5: R_Knee
        # 7: L_Ankle, 8: R_Ankle
        # 12: Neck
        # 16: L_Shoulder, 17: R_Shoulder
        # 18: L_Elbow, 19: R_Elbow
        
        def _get_joint_rot(joint_idx):
            # Extract the 3 axis-angle values for standard SMPL joints
            # Returns [x, y, z] rotation in radians
            start = joint_idx * 3
            end = start + 3
            if end <= len(thetas):
                return thetas[start:end]
            return np.array([0.0, 0.0, 0.0])

        out = {}
        
        # Root (Pelvis)
        # SMPL global orientation is joint 0
        root_rot = _get_joint_rot(0)
        # Assuming MuJoCo root_yaw is around Z, pitch around X, roll around Y
        out['root_yaw'] = root_rot[2]
        
        # Legs
        l_hip = _get_joint_rot(1)
        r_hip = _get_joint_rot(2)
        out['hip_pitch_l'] = l_hip[0]  # Simplified mapping
        out['hip_pitch_r'] = r_hip[0]
        
        l_knee = _get_joint_rot(4)
        r_knee = _get_joint_rot(5)
        # Knees in SMPL bend primarily on the X axis 
        out['knee_pitch_l'] = l_knee[0]
        out['knee_pitch_r'] = r_knee[0]

        l_ank = _get_joint_rot(7)
        r_ank = _get_joint_rot(8)
        out['ankle_pitch_l'] = l_ank[0]
        out['ankle_pitch_r'] = r_ank[0]

        # Torso (Spine integration)
        spine1 = _get_joint_rot(3)
        spine2 = _get_joint_rot(6)
        out['torso_pitch'] = spine1[0] + spine2[0]
        out['torso_roll'] = spine1[1] + spine2[1]

        # Arms
        l_shl = _get_joint_rot(16)
        r_shl = _get_joint_rot(17)
        out['shoulder_pitch_l'] = l_shl[0]
        out['shoulder_pitch_r'] = r_shl[0]
        out['shoulder_roll_l'] = l_shl[2]
        out['shoulder_roll_r'] = r_shl[2]

        l_elb = _get_joint_rot(18)
        r_elb = _get_joint_rot(19)
        out['elbow_pitch_l'] = l_elb[0]
        out['elbow_pitch_r'] = r_elb[0]

        # Neck
        neck = _get_joint_rot(12)
        out['neck_pitch'] = neck[0]
        out['neck_yaw'] = neck[2]

        # Scale limits intuitively
        for k in out:
            # Bound wildly out of scale axis-angle projections
            out[k] = float(np.clip(out[k], -3.14, 3.14))

        return out
