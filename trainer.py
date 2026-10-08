import cv2
from src.pose_detector import PoseDetector
from src.exercise_analyzer import SquatAnalyzer, BicepCurlAnalyzer, PushupAnalyzer
from src.voice_feedback import VoiceEngine

def main():
    detector = PoseDetector()
    voice = VoiceEngine()

    analyzers = {
        "1": ("SQUAT", SquatAnalyzer()),
        "2": ("BICEP_CURL", BicepCurlAnalyzer()),
        "3": ("PUSHUP", PushupAnalyzer())
    }

    selected_key = "1"  # Default exercise
    active_name, active_analyzer = analyzers[selected_key]

    cap = cv2.VideoCapture(0)
    print("FLEX AI TRAINER Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        keypoints, annotated_frame = detector.detect(frame)

        # On-Screen Menu Overlay Box
        cv2.rectangle(annotated_frame, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "SELECT EXERCISE:", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, "[1] Squat  [2] Bicep Curl  [3] Pushup", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"ACTIVE: {active_name}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if keypoints is not None:
            results = active_analyzer.process(keypoints)

            # Active Exercise Stats Display
            cv2.putText(annotated_frame, f"Reps: {results['counter']}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Stage: {results['stage']}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"{results['angle_name']}: {results['primary_angle']} deg", (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            color = (0, 0, 255) if results['status'] == "WARNING" else (0, 255, 0)
            cv2.putText(annotated_frame, results['feedback'], (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if results['status'] == "WARNING":
                voice.speak_async(results['feedback'])

        cv2.imshow("FLEX AI TRAINER", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in analyzers:
            selected_key = chr(key)
            active_name, active_analyzer = analyzers[selected_key]
            voice.speak_async(f"{active_name} selected")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
