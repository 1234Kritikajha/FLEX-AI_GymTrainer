import os
import numpy as np
from ultralytics import YOLO
import config

class PoseDetector:
    """
    YOLOv8-Pose Detector Wrapper with keypoint confidence extraction,
    safe error handling, and model fallback logic.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            if os.path.exists(config.YOLO_TRAINED_PATH):
                model_path = config.YOLO_TRAINED_PATH
            else:
                model_path = config.YOLO_PRETRAINED_PATH

        self.model_path = model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs YOLO pose detection on an input image/frame.
        Returns:
            keypoints: np.ndarray of shape (17, 3) containing [x, y, conf] for 17 COCO joints, or None
            annotated_frame: np.ndarray image with skeleton overlay
            pose_confidence: float overall confidence score
        """
        if frame is None:
            return None, frame, 0.0

        try:
            results = self.model(frame, verbose=False)
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kpt_obj = results[0].keypoints
                if hasattr(kpt_obj, 'data') and kpt_obj.data is not None and kpt_obj.data.shape[1] > 0:
                    kpts = kpt_obj.data[0].cpu().numpy()  # (17, 3)
                    annotated_frame = results[0].plot()
                    pose_conf = float(np.mean(kpts[:, 2]))
                    return kpts, annotated_frame, pose_conf
                elif hasattr(kpt_obj, 'xy') and kpt_obj.xy is not None and kpt_obj.xy.shape[1] > 0:
                    xy = kpt_obj.xy[0].cpu().numpy()  # (17, 2)
                    conf = np.ones((17, 1), dtype=np.float32)
                    kpts = np.hstack([xy, conf])
                    annotated_frame = results[0].plot()
                    return kpts, annotated_frame, 1.0
        except Exception as e:
            print(f"[PoseDetector] Exception during frame processing: {e}")

        return None, frame, 0.0
