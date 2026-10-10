# FLEX AI Gym Trainer — Technical System Documentation

**Version:** 2.0.0 (Hybrid Vision MVP)  
**Author:** AI System Architecture & Computer Vision Engineering Team  
**Technology Stack:** Python 3.14, PyTorch, Ultralytics YOLOv8, OpenCV, NumPy, PyTTSx3, Scikit-Learn  

---

## 1. Executive Summary

**FLEX AI Gym Trainer** is an edge-capable, real-time computer vision fitness coaching platform. The system processes live camera feeds to estimate human body poses, classify exercise movements, evaluate posture biomechanics, count repetitions via finite state machines (FSM), and deliver instant visual HUD and non-blocking auditory feedback.

Unlike conventional fitness applications that rely solely on heuristics or unconstrained black-box classification, FLEX AI employs a **Hybrid Decoupled Computer Vision Architecture**:
1. **Geometric Pose Estimation (YOLOv8-Pose)** extracts 17 COCO body keypoint coordinates and confidence metrics.
2. **Feature Preprocessing & Normalization** standardizes spatial coordinates relative to body scale and applies temporal exponential moving average (EMA) jitter smoothing.
3. **Deep Neural Exercise Classification (FlexCNN)** predicts the user's active exercise from normalized keypoint feature vectors.
4. **Multi-Factor Decision Verification Engine** validates whether the user's actual movement matches their selected exercise before authorizing rep counts.
5. **Deterministic Biomechanical Analyzers & FSMs** enforce movement safety angles, depth thresholds, and phase transitions.
6. **Asynchronous Non-Blocking Voice Engine** delivers audio feedback without stalling the real-time webcam frame processing pipeline.

---

## 2. System Architecture & Pipeline Flow

### Dataflow Pipeline Diagram

```
                        +----------------------------+
                        |     Live Video Feed        |
                        |      (Webcam / Stream)     |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |   YOLOv8-Pose Estimator    |
                        |  (17 COCO 2D Keypoints)    |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |    Keypoint Processor      |
                        |  - Torso Scale Normalization|
                        |  - Hip Centroid Translation|
                        |  - Confidence Filtering    |
                        |  - EMA Temporal Smoothing  |
                        +-------+------------+-------+
                                |            |
            +-------------------+            +-------------------+
            |                                                    |
            v                                                    v
+-----------------------+                            +-----------------------+
|  FlexCNN Classifier   |                            | Biomechanical Rules   |
| (1D Conv Neural Net)  |                            |  & Angle Evaluator    |
+-----------+-----------+                            +-----------+-----------+
            |                                                    |
            v                                                    v
+-----------------------+                            +-----------------------+
| Predicted Exercise    |                            | Motion State Machine  |
|  & Confidence %       |                            |  (FSM Rep Counter)    |
+-----------+-----------+                            +-----------+-----------+
            |                                                    |
            +-------------------+            +-------------------+
                                |            |
                                v            v
                        +----------------------------+
                        |      Decision Engine       |
                        | - Selected vs Detected     |
                        | - Confidence Thresholds    |
                        | - Rep Count Authorization  |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |      Real-Time HUD &       |
                        |  Asynchronous Speech Engine|
                        +----------------------------+
```

---

## 3. Core Component Breakdown

### 3.1 Pose Detector (`src/pose_detector.py`)
Wraps the Ultralytics YOLOv8-Pose model (`models/trained/flex_pose_best.pt` with fallback to `models/pretrained/yolov8n-pose.pt`).
* **Input:** Raw RGB video frame $(H \times W \times 3)$.
* **Output:** Keypoint array of shape $(17, 3)$ representing $[x, y, \text{confidence}]$ for 17 COCO anatomical joints, annotated rendering frame, and mean pose confidence score.
* **Resilience:** Traps inference exceptions and missing-person states, returning safe null representations without interrupting the live processing loop.

### 3.2 Keypoint Processor (`src/keypoint_processor.py`)
Ensures model invariance to camera distance, person height, and absolute bounding box position in the frame.
1. **Confidence Filtering:** Filters joints with confidence below `config.KEYPOINT_CONF_THRESHOLD` (0.25).
2. **Centroid Translation:** Translates spatial coordinates so the body center (mid-hip midpoint or mid-shoulder fallback) is aligned at $(0,0)$.
3. **Torso-Length Scale Normalization:** Scales coordinates by dividing by the Euclidean distance between mid-shoulder and mid-hip.
4. **Temporal EMA Smoothing:** Applies Exponential Moving Average ($\alpha = 0.3$) across consecutive frames to reduce high-frequency landmark jitter.
5. **Feature Mapping:** Converts normalized array $(17, 3)$ into Conv1D tensor shape $(3, 17)$ for PyTorch inference.

### 3.3 Neural Exercise Classifier — FlexCNN (`src/pose_classifier.py`)
Lightweight 1D Convolutional Neural Network optimized for real-time edge CPU/GPU execution.

```python
Input (Batch, 3, 17)
  │
  ├──► Conv1D(in=3, out=32, kernel=3, padding=1) ──► BatchNorm1d ──► ReLU
  ├──► MaxPool1d(kernel=2, stride=2, padding=1)
  ├──► Conv1D(in=32, out=64, kernel=3, padding=1) ──► BatchNorm1d ──► ReLU
  ├──► AdaptiveAvgPool1d(1)
  ├──► Linear(64 -> 32) ──► ReLU ──► Dropout(p=0.3)
  └──► Linear(32 -> 9) ──► Softmax Logits
```

* **Weights Checkpoint:** Saved at `models/trained/flex_cnn_best.pt`.
* **Class Mapping:** Machine-readable JSON dictionary at `models/trained/exercise_classes.json`.

### 3.4 Multi-Factor Decision Engine (`src/decision_engine.py`)
Acts as the central verification layer between neural predictions and physical movement logic:
* **Selected vs. Detected Verification:** Compares the exercise chosen in the UI against `FlexCNN` predictions.
* **Confidence Gate:** Enforces `CNN_CONF_THRESHOLD` (0.65). If confidence is lower or exercise mismatches, sets status to `EXERCISE_MISMATCH`, issues audio/visual warnings, and **disallows rep counting**.
* **Safety Rules Priority:** Neural classifications are prohibited from overriding biomechanical safety alerts.

### 3.5 Biomechanical Analyzers & FSM (`src/exercise_analyzer.py`)
Modular classes implementing exercise-specific angle calculations (`src/angle_utils.py`) and finite state machines:
* **Smart Side Selection:** Dynamically selects the left or right body side keypoint trios based on higher cumulative visibility confidence.
* **Standardized Result Schema:**
  ```json
  {
    "primary_angle": 105,
    "angle_name": "Knee Angle",
    "counter": 7,
    "stage": "BOTTOM",
    "feedback": "Good Squat Depth!",
    "status": "GOOD"
  }
  ```

### 3.6 Asynchronous Speech Engine (`src/voice_feedback.py`)
Integrates `pyttsx3` offline text-to-speech inside a background daemon thread. Implements cooldown debouncing (`config.VOICE_COOLDOWN = 2.5s`) to prevent speech overlap and eliminate UI loop blocking.

---

## 4. Supported Exercises & Biomechanical Threshold Rules

FLEX AI Trainer supports 9 canonical exercises with modular state machines:

| Exercise | Canonical Key | Primary Measured Angle | Joint Sequence | Motion Stages | Key Form Boundaries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Squat** | `squat` | Knee Angle | Hip - Knee - Ankle | `STANDING` $\rightarrow$ `DESCENDING` $\rightarrow$ `BOTTOM` $\rightarrow$ `ASCENDING` | Back Angle $> 120^\circ$; Bottom Knee $< 105^\circ$ |
| **Bicep Curl** | `bicep_curl` | Elbow Angle | Shoulder - Elbow - Wrist | `DOWN` $\rightarrow$ `UP` $\rightarrow$ `DOWN` | Extended $> 145^\circ$; Flexed $< 50^\circ$ |
| **Push-Up** | `pushup` | Elbow Angle | Shoulder - Elbow - Wrist | `UP` $\rightarrow$ `DOWN` $\rightarrow$ `UP` | Body Line $> 140^\circ$; Bottom Elbow $< 100^\circ$ |
| **Lunge** | `lunge` | Front Knee Angle | Hip - Knee - Ankle | `STANDING` $\rightarrow$ `DOWN` $\rightarrow$ `BOTTOM` $\rightarrow$ `UP` | Extended $> 155^\circ$; Depth $< 100^\circ$ |
| **Plank** | `plank` | Body Line Angle | Shoulder - Hip - Ankle | `PLANK_HOLD` (Hold Timer) | Straight Line $140^\circ \le \text{Angle} \le 175^\circ$ |
| **Glute Bridge** | `glute_bridge` | Hip Extension | Shoulder - Hip - Knee | `DOWN` $\rightarrow$ `UP` $\rightarrow$ `DOWN` | Peak Extension $> 155^\circ$; Return $< 120^\circ$ |
| **Crunch** | `crunch` | Torso Flex Angle | Shoulder - Hip - Knee | `DOWN` $\rightarrow$ `UP` $\rightarrow$ `DOWN` | Curl Flexion $< 110^\circ$; Extension $> 140^\circ$ |
| **Shoulder Press** | `shoulder_press` | Overhead Extension | Shoulder - Elbow - Wrist | `DOWN` $\rightarrow$ `UP` $\rightarrow$ `DOWN` | Lockout $> 150^\circ$; Down Depth $< 90^\circ$ |
| **Tricep Extension**| `tricep_extension`| Arm Extension | Shoulder - Elbow - Wrist | `BENT` $\rightarrow$ `EXTENDED` $\rightarrow$ `BENT` | Full Extension $> 150^\circ$; Flexed $< 85^\circ$ |

---

## 5. System Configuration Schema (`config.py`)

All global hyper-parameters, file system paths, and confidence limits are centrally configured:

```python
# Model Directories & Checkpoint Paths
PRETRAINED_MODELS_DIR = "models/pretrained"
TRAINED_MODELS_DIR    = "models/trained"
YOLO_TRAINED_PATH     = "models/trained/flex_pose_best.pt"
CNN_TRAINED_PATH      = "models/trained/flex_cnn_best.pt"
CLASS_MAPPING_PATH    = "models/trained/exercise_classes.json"

# Operational Thresholds
YOLO_CONF_THRESHOLD      = 0.35  # Minimum YOLO person detection threshold
KEYPOINT_CONF_THRESHOLD  = 0.25  # Minimum individual joint confidence
CNN_CONF_THRESHOLD       = 0.65  # Minimum exercise classifier verification confidence
TEMPORAL_SMOOTHING_ALPHA = 0.30  # EMA factor (0.0 = static, 1.0 = raw)
VOICE_COOLDOWN           = 2.50  # Seconds between voice alerts
```

---

## 6. Dataset Lifecycle & Training Pipeline

The project includes an automated end-to-end data generation, preprocessing, training, and validation workflow under `training/`.

### 6.1 Automated Dataset Generation & Splitting (`prepare_cnn_dataset.py`)
Generates 9,000 spatial keypoint frame sequences across all 9 canonical exercises with physics-informed joint kinematics, camera perspective variation, height scaling, and sensor noise.
* **Anti-Data-Leakage Splitting:** Splits data at the sequence/session level rather than randomly permuting adjacent frames:
  * **Train Set (70%):** 6,300 samples
  * **Validation Set (15%):** 1,350 samples
  * **Test Set (15%):** 1,350 samples

### 6.2 Neural Model Training (`train_cnn.py`)
Trains `FlexCNN` using `AdamW` optimizer, `CrossEntropyLoss`, and `ReduceLROnPlateau` scheduler over 35 epochs.
* Automatically selects hardware accelerator (`mps` for Apple Silicon GPU, `cuda` for Nvidia GPU, or `cpu`).
* Saves best checkpoint to `models/trained/flex_cnn_best.pt` and loss curves to `outputs/training/cnn/learning_curves.png`.

### 6.3 Evaluation & Metrics (`validate_cnn.py`)
Evaluates the model on unseen test data and outputs full classification metrics:
* **Test Accuracy:** **85.85%**
* **Weighted Precision:** `0.8914`
* **Weighted Recall:** `0.8585`
* **Weighted F1 Score:** `0.8550`

### 6.4 Orchestrator Script (`training/train_all.py`)
Runs the full pipeline sequentially:
```bash
PYTHONPATH=. .venv/bin/python training/train_all.py
```

---

## 7. Quality Assurance & Automated Test Suite

An automated test suite (`test_pipeline.py`) validates system functionality across 13 test criteria:

```text
======================================================================
RUNNING AUTOMATED VERIFICATION TEST SUITE (TESTS 1 - 13)
======================================================================
[TEST 1/13]  Importing all system modules                     --> PASSED
[TEST 2/13]  Loading YOLO pose detector                       --> PASSED
[TEST 3/13]  Loading FlexCNN trained classifier               --> PASSED
[TEST 4/13]  Running keypoint processor normalization         --> PASSED
[TEST 5/13]  Running CNN exercise prediction                  --> PASSED
[TEST 6/13]  Evaluating Decision Engine                       --> PASSED
[TEST 7/13]  Validating Exercise Analyzers (9/9)              --> PASSED
[TEST 8/13]  Checking trainer.py entrypoint syntax            --> PASSED
[TEST 9/13]  Testing no-person-detected safety (None kpts)    --> PASSED
[TEST 10/13] Testing low-confidence keypoints safety          --> PASSED
[TEST 11/13] Testing exercise mismatch detection               --> PASSED
[TEST 12/13] Testing exercise match rep authorization         --> PASSED
[TEST 13/13] Testing non-blocking voice feedback engine       --> PASSED
======================================================================
ALL 13 AUTOMATED PIPELINE TESTS PASSED SUCCESSFULLY!
======================================================================
```

Execute tests via:
```bash
PYTHONPATH=. .venv/bin/python test_pipeline.py
```

---

## 8. Directory & Repository Structure

```text
FLEX-AI_GymTrainer/
├── config.py                         # Global hyper-parameters & system config
├── trainer.py                        # Main runtime application & HUD interface
├── test_pipeline.py                  # Automated 13-point test suite
├── DOCUMENTATION.md                  # Comprehensive technical documentation
├── README.md                         # Quick-start documentation
├── PROJECT_STATUS.md                 # Project release status & metrics
├── requirements.txt                  # Python dependencies
├── models/
│   ├── pretrained/
│   │   └── yolov8n-pose.pt           # Base Ultralytics YOLOv8 pose weights
│   └── trained/
│       ├── flex_pose_best.pt         # Fine-tuned YOLO pose weights
│       ├── flex_cnn_best.pt          # PyTorch FlexCNN classifier weights
│       └── exercise_classes.json     # Class mapping configuration
├── src/
│   ├── angle_utils.py                # Trigonometric 2D angle utilities
│   ├── decision_engine.py            # Multi-factor decision & verification layer
│   ├── exercise_analyzer.py          # 9 exercise analyzers & factory registry
│   ├── keypoint_processor.py         # Normalization, translation & EMA smoothing
│   ├── pose_classifier.py            # PyTorch FlexCNN model & inference wrapper
│   ├── pose_detector.py              # YOLOv8-Pose wrapper
│   └── voice_feedback.py             # Asynchronous debounced TTS voice engine
├── dataset/
│   ├── data.yaml                     # YOLO pose dataset configuration
│   └── pose_classifier/              # Serialized PyTorch feature datasets
├── training/
│   ├── prepare_dataset.py            # YOLO pose image dataset inspector
│   ├── prepare_cnn_dataset.py        # CNN keypoint feature generator & splitter
│   ├── train_pose.py                 # YOLO pose fine-tuning script
│   ├── train_cnn.py                  # PyTorch FlexCNN model trainer
│   ├── validate_cnn.py               # Classification performance evaluator
│   ├── test_cnn.py                   # Single sample inference tester
│   └── train_all.py                  # Pipeline master orchestrator
└── outputs/
    └── training/
        ├── cnn/                      # Learning curves & confusion matrices
        └── yolo/                     # YOLO pose training logs
```

---

## 9. Operating Guide

### 9.1 Environment Setup
```bash
cd FLEX-AI_GymTrainer
source .venv/bin/activate
pip install -r requirements.txt
```

### 9.2 Running Live Application
```bash
PYTHONPATH=. .venv/bin/python trainer.py
```

### 9.3 In-App Controls
* **Keys `1` – `9`**: Switch active exercise target.
* **Key `q` or `ESC`**: Exit application.

---

## 10. Developer Guide: Adding a New Exercise

To extend the system with a 10th exercise (e.g. `lateral_raise`):

1. **Update Canonical List (`config.py`):**
   Add `"lateral_raise"` to `CANONICAL_EXERCISES` and define display name and stages.
2. **Implement Analyzer Class (`src/exercise_analyzer.py`):**
   ```python
   class LateralRaiseAnalyzer(BaseAnalyzer):
       def __init__(self):
           super().__init__("lateral_raise")
           self.stage = "DOWN"

       def process(self, kpts):
           # Calculate shoulder abduct angle
           ...
           return {
               "primary_angle": int(abduct_angle),
               "angle_name": "Arm Lift Angle",
               "counter": self.counter,
               "stage": self.stage,
               "feedback": feedback,
               "status": status
           }
   ```
3. **Register in Factory (`src/exercise_analyzer.py`):**
   Add `"lateral_raise": LateralRaiseAnalyzer` to `ExerciseAnalyzerFactory._analyzers`.
4. **Add Kinematics Generator (`training/prepare_cnn_dataset.py`):**
   Add arm abduction motion equations in `generate_synthetic_exercise_keypoints()`.
5. **Retrain & Validate:**
   Run `PYTHONPATH=. .venv/bin/python training/train_all.py`.
