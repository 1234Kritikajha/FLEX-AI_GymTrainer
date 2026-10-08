import os
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path='models/pretrained/yolov8n-pose.pt'):
        # Fallback if custom trained model exists
        if os.path.exists('models/trained/flex_pose_best.pt'):
            model_path = 'models/trained/flex_pose_best.pt'
        
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        if len(results[0].keypoints) > 0 and results[0].keypoints.xy.shape[1] > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            annotated_frame = results[0].plot()
            return keypoints, annotated_frame
        return None, frame
