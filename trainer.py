import cv2
import config
from src.pose_detector import PoseDetector
from src.keypoint_processor import KeypointProcessor
from src.pose_classifier import PoseClassifier
from src.decision_engine import DecisionEngine
from src.exercise_analyzer import ExerciseAnalyzerFactory
from src.voice_feedback import VoiceEngine

def main():
    print("=" * 60)
    print("STARTING FLEX AI TRAINER (HYBRID POSE / EXERCISE CLASSIFIER)")
    print("=" * 60)

    detector = PoseDetector()
    processor = KeypointProcessor(enable_smoothing=True)
    classifier = PoseClassifier()
    decision_engine = DecisionEngine()
    voice = VoiceEngine()

    # Exercise Menu Registry (Keys 1-9)
    menu_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    canonical_list = config.CANONICAL_EXERCISES

    analyzers = {
        str(i + 1): (
            canonical_list[i],
            config.EXERCISE_DISPLAY_NAMES[canonical_list[i]],
            ExerciseAnalyzerFactory.get_analyzer(canonical_list[i])
        )
        for i in range(len(canonical_list))
    }

    selected_key = "1"
    active_canonical, active_display, active_analyzer = analyzers[selected_key]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (video device 0).")
        return

    print("FLEX AI TRAINER Webcam Stream Started...")
    print("Controls: Press keys [1-9] to switch exercise. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # 1. Pose Detection
        raw_keypoints, annotated_frame, pose_conf = detector.detect(frame)

        # 2. Keypoint Preprocessing & CNN Classification
        cnn_result = classifier.predict(raw_keypoints)

        # 3. Biomechanical Rule Analysis
        if raw_keypoints is not None:
            bio_result = active_analyzer.process(raw_keypoints)
        else:
            bio_result = {
                "primary_angle": 0,
                "angle_name": "Angle",
                "counter": active_analyzer.counter,
                "stage": "NO_POSE",
                "feedback": "Position full body in camera frame",
                "status": "NO_POSE"
            }

        # 4. Decision Engine Verification
        decision = decision_engine.evaluate(
            selected_exercise=active_canonical,
            cnn_result=cnn_result,
            pose_confidence=pose_conf,
            biomechanics_result=bio_result
        )

        # 5. On-Screen UI Overlay HUD
        # Top Menu Banner
        cv2.rectangle(annotated_frame, (10, 10), (630, 95), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "FLEX AI TRAINER — EXERCISE SELECTOR [1-9]:", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, "[1]Squat [2]Bicep [3]Pushup [4]Lunge [5]Plank", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(annotated_frame, "[6]Bridge [7]Crunch [8]Press [9]Tricep", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"ACTIVE: {active_display.upper()}", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Real-Time Diagnostics Panel
        cv2.rectangle(annotated_frame, (10, 105), (380, 310), (0, 0, 0), -1)

        detected_name = config.EXERCISE_DISPLAY_NAMES.get(
            decision['detected_exercise'], decision['detected_exercise']
        ) if decision['detected_exercise'] else "None"

        cnn_pct = int(decision['cnn_confidence'] * 100)
        pose_pct = int(decision['pose_confidence'] * 100)

        cv2.putText(annotated_frame, f"SELECTED: {active_display.upper()}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"DETECTED: {detected_name.upper()} ({cnn_pct}%)", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if decision['status'] != "EXERCISE_MISMATCH" else (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"POSE CONF: {pose_pct}%", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        counter_label = "Hold Time" if active_canonical == "plank" else "Reps"
        unit = "s" if active_canonical == "plank" else ""
        cv2.putText(annotated_frame, f"{counter_label}: {bio_result['counter']}{unit}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Stage: {bio_result['stage']}", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"{bio_result['angle_name']}: {bio_result['primary_angle']} deg", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Status & Feedback Display
        if decision['status'] == "EXERCISE_MISMATCH":
            status_color = (0, 0, 255)  # Red
            cv2.putText(annotated_frame, "STATUS: EXERCISE MISMATCH", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            voice.speak_async("Exercise mismatch. Please perform the selected exercise.")
        elif decision['status'] == "WARNING":
            status_color = (0, 165, 255)  # Orange
            cv2.putText(annotated_frame, f"STATUS: FORM WARNING", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            voice.speak_async(decision['feedback'])
        elif decision['status'] == "NO_POSE":
            status_color = (128, 128, 128)  # Gray
            cv2.putText(annotated_frame, "STATUS: SEARCHING FOR POSE", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        else:
            status_color = (0, 255, 0)  # Green
            cv2.putText(annotated_frame, f"STATUS: {decision['status']}", (20, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Feedback Text Banner
        cv2.putText(annotated_frame, decision['feedback'], (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("FLEX AI TRAINER", annotated_frame)

        # Keypress Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif chr(key) in analyzers:
            selected_key = chr(key)
            active_canonical, active_display, active_analyzer = analyzers[selected_key]
            active_analyzer.reset()
            voice.speak_async(f"{active_display} selected")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
