# FLEX AI Trainer — Hybrid Vision AI & Exercise Coach

FLEX AI Trainer is a real-time computer vision fitness coaching application. It combines **YOLOv8-Pose**, a custom lightweight **Conv1D Neural Network (FlexCNN)**, interpretable **biomechanical angle rules**, **Finite State Machines (FSM)** for rep counting, and **asynchronous voice feedback**.

> 📖 **Full Technical Documentation:** For comprehensive architecture diagrams, component specifications, and developer guides, see [DOCUMENTATION.md](file:///Users/navinkumarjha/Documents/Personal%20Gym%20Trainer/FLEX-AI_GymTrainer/DOCUMENTATION.md).

---

## Architecture Overview

```text
WEBCAM
   ↓
YOLOv8-Pose (17 COCO Keypoints)
   ↓
Keypoint Preprocessing (Confidence Filtering, Body Normalization, EMA Smoothing)
   ↓
FlexCNN Exercise Classifier (Conv1D Neural Network)
   ↓
Decision Engine (Selected vs. Detected Exercise Verification, Thresholding)
   ↓
Biomechanical Rules & FSM (Form Evaluation & Rep Counting)
   ↓
Text & Non-Blocking Voice Feedback (pyttsx3 Asynchronous Engine)
```

### Module Responsibilities
* **YOLOv8-Pose**: Responsible strictly for 2D human pose detection and 17 body keypoint extraction.
* **Keypoint Processor**: Filters low-confidence joints, translates coordinates relative to body center, normalizes scale by torso length, and applies temporal smoothing.
* **FlexCNN**: Recognizes exercise identity from normalized 17-keypoint sequences. Does NOT directly generate natural language.
* **Decision Engine**: Verifies that the user's movement matches the selected exercise before permitting rep counting or exercise-specific feedback.
* **Biomechanics + FSM**: Interpretable geometry rules compute joint angles, enforce form boundaries, and track movement phase transitions for rep validation.
* **Voice Engine**: Delivers non-blocking audio alerts with cooldown debouncing.

---

## Supported Exercises (9 Canonical Movements)

1. **Squat** (`squat`) — Knee & back angle state machine (`STANDING`, `DESCENDING`, `BOTTOM`, `ASCENDING`).
2. **Bicep Curl** (`bicep_curl`) — Elbow angle tracking (`DOWN`, `UP`).
3. **Push-Up** (`pushup`) — Elbow depth & body alignment check (`UP`, `DOWN`).
4. **Lunge** (`lunge`) — Front knee angle tracking (`STANDING`, `DOWN`, `BOTTOM`, `UP`).
5. **Plank** (`plank`) — Body line alignment & hold duration timer (`PLANK_HOLD`).
6. **Glute Bridge** (`glute_bridge`) — Hip extension tracking (`DOWN`, `UP`).
7. **Crunch** (`crunch`) — Torso flexion angle tracking (`DOWN`, `UP`).
8. **Shoulder Press** (`shoulder_press`) — Overhead elbow extension tracking (`DOWN`, `UP`).
9. **Tricep Extension** (`tricep_extension`) — Elbow bend to extension tracking (`BENT`, `EXTENDED`).

---

## Model Locations

* **Pretrained YOLO Model**: `models/pretrained/yolov8n-pose.pt`
* **Trained Pose Model**: `models/trained/flex_pose_best.pt`
* **Trained CNN Model**: `models/trained/flex_cnn_best.pt`
* **Class Mapping Config**: `models/trained/exercise_classes.json`

---

## Setup & Running

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python trainer.py
```

### Keyboard Controls
* Press `1` – `9` to switch active exercise:
  * `[1]` Squat
  * `[2]` Bicep Curl
  * `[3]` Push-Up
  * `[4]` Lunge
  * `[5]` Plank
  * `[6]` Glute Bridge
  * `[7]` Crunch
  * `[8]` Shoulder Press
  * `[9]` Tricep Extension
* Press `q` or `ESC` to quit.

---

## Retraining & Pipeline Orchestration

You can execute the automated retraining & validation pipeline:

```bash
python training/train_all.py
```

Or run individual pipeline steps:

1. **Inspect Pose Dataset**: `python training/prepare_dataset.py`
2. **Train Pose Model**: `python training/train_pose.py`
3. **Validate Pose Model**: `python training/validate_pose.py`
4. **Prepare CNN Features**: `python training/prepare_cnn_dataset.py`
5. **Train CNN Classifier**: `python training/train_cnn.py`
6. **Validate CNN Classifier**: `python training/validate_cnn.py`
7. **Test CNN Inference**: `python training/test_cnn.py`
8. **Run Verification Test Suite**: `python test_pipeline.py`

---

## How to Add a New Exercise

1. **Add Canonical Name**: Add the new exercise identifier (e.g. `lateral_raise`) to `CANONICAL_EXERCISES` in `config.py`.
2. **Create Analyzer Class**: Subclass `BaseAnalyzer` in `src/exercise_analyzer.py` implementing `process(self, kpts)`.
3. **Register in Factory**: Register your analyzer class in `ExerciseAnalyzerFactory._analyzers` in `src/exercise_analyzer.py`.
4. **Add Kinematics to Dataset Prep**: Update `generate_synthetic_exercise_keypoints()` in `training/prepare_cnn_dataset.py` with joint kinematics.
5. **Retrain CNN**: Run `python training/train_all.py` to retrain `flex_cnn_best.pt`.
