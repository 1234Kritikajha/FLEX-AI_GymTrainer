import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRETRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "pretrained")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Model Paths
YOLO_PRETRAINED_PATH = os.path.join(PRETRAINED_MODELS_DIR, "yolov8n-pose.pt")
YOLO_TRAINED_PATH = os.path.join(TRAINED_MODELS_DIR, "flex_pose_best.pt")
CNN_TRAINED_PATH = os.path.join(TRAINED_MODELS_DIR, "flex_cnn_best.pt")
CLASS_MAPPING_PATH = os.path.join(TRAINED_MODELS_DIR, "exercise_classes.json")

# Ensure required directories exist
for path in [PRETRAINED_MODELS_DIR, TRAINED_MODELS_DIR, OUTPUTS_DIR, 
            os.path.join(OUTPUTS_DIR, "training", "yolo"),
            os.path.join(OUTPUTS_DIR, "training", "cnn"),
            os.path.join(OUTPUTS_DIR, "logs"),
            DATASET_DIR, os.path.join(DATASET_DIR, "pose_classifier")]:
    os.makedirs(path, exist_ok=True)

# Thresholds & Parameters
YOLO_CONF_THRESHOLD = 0.35
KEYPOINT_CONF_THRESHOLD = 0.25
CNN_CONF_THRESHOLD = 0.65
TEMPORAL_SMOOTHING_ALPHA = 0.3  # Exponential Moving Average factor (0.0 to 1.0)

# Voice Feedback
VOICE_COOLDOWN = 2.5  # Seconds between spoken feedback

# Supported Canonical Exercises
CANONICAL_EXERCISES = [
    "squat",
    "bicep_curl",
    "pushup",
    "lunge",
    "plank",
    "glute_bridge",
    "crunch",
    "shoulder_press",
    "tricep_extension"
]

EXERCISE_DISPLAY_NAMES = {
    "squat": "Squat",
    "bicep_curl": "Bicep Curl",
    "pushup": "Push-Up",
    "lunge": "Lunge",
    "plank": "Plank",
    "glute_bridge": "Glute Bridge",
    "crunch": "Crunch",
    "shoulder_press": "Shoulder Press",
    "tricep_extension": "Tricep Extension"
}

# Exercise Stages
EXERCISE_STAGES = {
    "squat": ["STANDING", "DESCENDING", "BOTTOM", "ASCENDING"],
    "bicep_curl": ["DOWN", "UP"],
    "pushup": ["UP", "DOWN"],
    "lunge": ["STANDING", "DOWN", "BOTTOM", "UP"],
    "plank": ["PLANK_HOLD"],
    "glute_bridge": ["DOWN", "UP"],
    "crunch": ["DOWN", "UP"],
    "shoulder_press": ["DOWN", "UP"],
    "tricep_extension": ["BENT", "EXTENDED"]
}

# COCO 17 Keypoints Map
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}
