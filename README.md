<<<<<<< HEAD
# FLEX AI Trainer — Full Body Exercise Coach

## What this version adds

- English-only voice instructions and on-screen coaching.
- Full-body YOLOv8-Pose skeleton with 17 keypoints.
- Exercise selector:
  1. Squat
  2. Bicep Curl
  3. Push-up
  4. Tricep Extension
  5. Glute Kickback
  6. Crunch
  7. Plank
  8. Hip Bridge
- Exercise-specific movement state machines.
- Rep counting for the rep-based exercises.
- Plank hold timer instead of fake rep counting.
- Exercise-specific setup instructions and form feedback.
- Optional Gemini English coaching.
- Gemini never controls the rep counter.
- Offline pyttsx3 coaching still works without Gemini.

## Important accuracy note

This is an MVP based on 2D webcam pose geometry. Accuracy depends heavily on camera angle,
lighting, visibility of the full body, clothing, occlusion, and exercise technique.

For best results:
- Keep the entire body inside the camera frame.
- Use the recommended camera angle shown in the application.
- Avoid multiple people in the frame.
- Keep the camera stable.
- For floor exercises, use a side view.

The current version uses deterministic angle/state rules rather than claiming that an
untrained CNN/LSTM model is already available.

## Setup

### 1. Enter the project folder

```bash
cd FLEX_AI_TRAINER_FULL_BODY
```

### 2. Create/activate your virtual environment

Example:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Put YOLO model in the project folder

Make sure this file exists:

```text
yolov8n-pose.pt
```

If your existing project already has it, copy it here.

### 5. Configure Gemini (optional)

```bash
cp .env.example .env
```

Then put your NEW Gemini API key in `.env`:

```env
GEMINI_API_KEY=YOUR_NEW_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
```

Never commit `.env` to Git.

If Gemini is unavailable, FLEX still works with offline English coaching.

## Run

```bash
python trainer.py
```

Choose an exercise from the menu.

### Controls

- `Q` = return to exercise menu / quit
- `ESC` = quit current exercise

## Exercise counting

### Squat
Tracks knee angle through down -> bottom -> up -> standing.

### Bicep Curl
Tracks elbow angle through extension -> curl -> extension.

### Push-up
Tracks elbow bend -> bottom -> full extension.

### Tricep Extension
Tracks elbow bend -> extension.

### Glute Kickback
Uses side-view hip movement to detect kickback -> return.

### Crunch
Uses side-view hip/torso geometry to detect crunch -> return.

### Hip Bridge
Tracks hip angle through bridge-up -> controlled return.

### Plank
A plank is time-based, so FLEX reports hold time and best hold time rather than inventing reps.

## Security

If an API key has ever been pasted into a chat, source file, screenshot, or Git repository,
treat it as exposed. Revoke/rotate it and create a new key.

Do not put secrets into `trainer.py`.
=======
# GYM-Trainer
>>>>>>> fc810b68cc8e3c03b9596e9acb6f0fca34a3865d
