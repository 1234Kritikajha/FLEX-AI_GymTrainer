import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config
from src.keypoint_processor import KeypointProcessor

class FlexCNN(nn.Module):
    """
    Lightweight Conv1D neural network operating on 17 normalized body keypoints.
    Input shape: (batch_size, 3, 17) -> [norm_x, norm_y, conf] across 17 COCO joints.
    """
    def __init__(self, num_classes=len(config.CANONICAL_EXERCISES)):
        super(FlexCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, 3, 17)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

class PoseClassifier:
    """
    Inference wrapper for FlexCNN exercise classifier model.
    """
    def __init__(self, model_path=config.CNN_TRAINED_PATH, class_mapping_path=config.CLASS_MAPPING_PATH):
        self.processor = KeypointProcessor(enable_smoothing=True)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Default class mapping fallback
        self.classes = config.CANONICAL_EXERCISES
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                mapping = json.load(f)
                self.idx_to_class = {int(k): v for k, v in mapping.get("idx_to_class", {}).items()}
                self.classes = list(self.idx_to_class.values())

        self.model = FlexCNN(num_classes=len(self.classes)).to(self.device)
        self.is_loaded = False

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.is_loaded = True
            except Exception as e:
                print(f"[PoseClassifier] Warning loading weights: {e}")
        else:
            print(f"[PoseClassifier] Notice: Model weights not found at {model_path}. Run training first.")

    def predict(self, raw_keypoints):
        """
        Runs keypoints through processor and CNN classifier.
        Returns dictionary containing predicted exercise, confidence, and probabilities.
        """
        if raw_keypoints is None or not self.is_loaded:
            return {
                "exercise": None,
                "confidence": 0.0,
                "stage": None,
                "probabilities": {}
            }

        norm_kpts = self.processor.process(raw_keypoints, is_live=True)
        if norm_kpts is None:
            return {
                "exercise": None,
                "confidence": 0.0,
                "stage": None,
                "probabilities": {}
            }

        conv1d_input = self.processor.get_conv1d_input(norm_kpts)
        tensor_in = torch.from_numpy(conv1d_input).unsqueeze(0).to(self.device)  # (1, 3, 17)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        pred_exercise = self.idx_to_class.get(top_idx, "unknown")
        confidence = float(probs[top_idx])

        prob_dict = {self.idx_to_class[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "exercise": pred_exercise,
            "confidence": confidence,
            "stage": None,  # Stage handled by biomechanics / FSM
            "probabilities": prob_dict
        }
