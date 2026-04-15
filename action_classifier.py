import numpy as np
import time
from collections import deque

# YOLO COCO format indices
NOSE = 0
L_SHL = 5; R_SHL = 6
L_ELB = 7; R_ELB = 8
L_WRI = 9; R_WRI = 10
L_HIP = 11; R_HIP = 12
L_KNE = 13; R_KNE = 14
L_ANK = 15; R_ANK = 16

class ActionClassifier:
    def __init__(self):
        self.state = "STANDING"
        self.wrist_x_history = deque(maxlen=20)
        self.last_update = time.time()
        
    def _is_visible(self, kpts, *indices, conf=0.5):
        for idx in indices:
            if kpts[idx][2] < conf:
                return False
        return True

    def classify(self, kpts) -> str:
        """
        Takes (17, 3) numpy array from YOLO.
        Returns the classified action string.
        """
        if kpts is None:
            return self.state
            
        # 1. ARMS_OUT (T-Pose)
        # Check if wrists are horizontally extremely wide compared to shoulders
        if self._is_visible(kpts, L_WRI, R_WRI, L_SHL, R_SHL, conf=0.3):
            shl_width = abs(kpts[L_SHL][0] - kpts[R_SHL][0])
            if shl_width > 10:
                l_dist = abs(kpts[L_WRI][0] - kpts[L_SHL][0])
                r_dist = abs(kpts[R_WRI][0] - kpts[R_SHL][0])
                if l_dist > shl_width * 1.2 and r_dist > shl_width * 1.2:
                    self.state = "ARMS_OUT"
                    return self.state

        # 2. WAVING (Arm Raised)
        # Check if EITHER wrist is raised physically above the shoulder
        # In image coords, +Y is DOWN, so smaller Y means higher.
        if self._is_visible(kpts, R_WRI, R_SHL, conf=0.4):
            if kpts[R_WRI][1] < kpts[R_SHL][1]:
                self.state = "WAVING"
                return self.state
                
        if self._is_visible(kpts, L_WRI, L_SHL, conf=0.4):
            if kpts[L_WRI][1] < kpts[L_SHL][1]:
                self.state = "WAVING"
                return self.state

        # 3. SQUATTING
        # Check if shoulders have dropped significantly closer to knees
        if self._is_visible(kpts, L_KNE, R_KNE, L_SHL, R_SHL, conf=0.4):
            shl_y = (kpts[L_SHL][1] + kpts[R_SHL][1]) / 2.0
            kne_y = (kpts[L_KNE][1] + kpts[R_KNE][1]) / 2.0
            shl_width = abs(kpts[L_SHL][0] - kpts[R_SHL][0])
            
            if shl_width > 10:
                # height between shoulders and knees
                vert_dist = kne_y - shl_y
                # If the vertical distance is extremely small compared to shoulder width,
                # the person is crouching/squatting down.
                if vert_dist > 0 and vert_dist < (shl_width * 1.8):
                    self.state = "SQUATTING"
                    return self.state

        # Default Fallback
        self.state = "STANDING"
        return self.state
