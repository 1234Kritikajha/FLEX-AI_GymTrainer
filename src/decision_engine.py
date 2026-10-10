import config

class DecisionEngine:
    """
    Combines selected exercise verification, CNN classification confidence,
    YOLO keypoint detection confidence, biomechanical rules, and FSM stage state.
    """
    def __init__(self, cnn_conf_threshold=config.CNN_CONF_THRESHOLD, pose_conf_threshold=config.YOLO_CONF_THRESHOLD):
        self.cnn_conf_threshold = cnn_conf_threshold
        self.pose_conf_threshold = pose_conf_threshold

    def evaluate(self, selected_exercise, cnn_result, pose_confidence, biomechanics_result):
        """
        Evaluates system inputs and returns decision dictionary:
        {
            "status": "GOOD" | "WARNING" | "EXERCISE_MISMATCH" | "NO_POSE",
            "feedback": str,
            "is_rep_allowed": bool,
            "detected_exercise": str,
            "cnn_confidence": float,
            "pose_confidence": float
        }
        """
        selected_clean = selected_exercise.lower().strip()
        detected_exercise = cnn_result.get("exercise") if cnn_result else None
        cnn_confidence = cnn_result.get("confidence", 0.0) if cnn_result else 0.0

        # 1. Pose Confidence Check
        if pose_confidence < self.pose_conf_threshold:
            return {
                "status": "NO_POSE",
                "feedback": "Position yourself clearly in front of the camera.",
                "is_rep_allowed": False,
                "detected_exercise": detected_exercise,
                "cnn_confidence": cnn_confidence,
                "pose_confidence": pose_confidence
            }

        # 2. Exercise Verification via CNN Classifier
        if detected_exercise is not None and cnn_confidence >= self.cnn_conf_threshold:
            if detected_exercise != selected_clean:
                display_selected = config.EXERCISE_DISPLAY_NAMES.get(selected_clean, selected_clean)
                display_detected = config.EXERCISE_DISPLAY_NAMES.get(detected_exercise, detected_exercise)
                return {
                    "status": "EXERCISE_MISMATCH",
                    "feedback": f"Please perform the selected {display_selected} exercise. (Detected: {display_detected})",
                    "is_rep_allowed": False,
                    "detected_exercise": detected_exercise,
                    "cnn_confidence": cnn_confidence,
                    "pose_confidence": pose_confidence
                }

        # 3. Biomechanics and Form Verification
        bio_status = biomechanics_result.get("status", "GOOD") if biomechanics_result else "GOOD"
        bio_feedback = biomechanics_result.get("feedback", "Good Form") if biomechanics_result else "Good Form"

        return {
            "status": bio_status,
            "feedback": bio_feedback,
            "is_rep_allowed": True,
            "detected_exercise": detected_exercise or selected_clean,
            "cnn_confidence": cnn_confidence,
            "pose_confidence": pose_confidence
        }
