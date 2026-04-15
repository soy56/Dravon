import numpy as np
from ultralytics import YOLO

class YoloTracker:
    def __init__(self, model_path='yolo11n-pose.pt', conf=0.5):
        """
        Wrapper for Ultralytics YOLO-Pose.
        Downloads and loads the ultra-fast Nano pose model.
        """
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.conf = conf
        print("  [OK] YOLO-Pose initialized")

    def process(self, frame_bgr):
        """
        Process a BGR frame and return the raw YOLO results.
        Runs entirely locally without internet after initial download.
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            verbose=False,
            device='cpu' # Ensure it runs deterministically
        )
        return results[0] if len(results) > 0 else None

    @staticmethod
    def get_keypoints(result):
        """
        Extracts the first detected human's keypoints as a numpy array.
        Shape: (17, 3) -> [x, y, confidence]
        Returns None if no human is detected.
        """
        if result and result.keypoints and result.keypoints.data.shape[1] == 17:
            # We take the first person detected
            return result.keypoints.data[0].cpu().numpy()
        return None
