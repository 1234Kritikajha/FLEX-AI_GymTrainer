from src.angle_utils import calculate_angle

class SquatAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "STANDING" # States: STANDING -> DESCENDING -> BOTTOM -> ASCENDING

    def process(self, kpts):
        # COCO 17 Keypoints Index Mapping:
        # 5: L_Shoulder, 11: L_Hip, 13: L_Knee, 15: L_Ankle
        shoulder = kpts[5]
        hip = kpts[11]
        knee = kpts[13]
        ankle = kpts[15]

        # Angles
        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)

        feedback = "Ready"
        status = "GOOD"

        # Back check
        if back_angle < 130:
            feedback = "Keep Back Straight!"
            status = "WARNING"

        # Movement state machine
        if knee_angle > 160:
            if self.stage == "ASCENDING":
                self.counter += 1
                feedback = f"Rep {self.counter} Complete!"
            self.stage = "STANDING"

        elif knee_angle < 100:
            self.stage = "BOTTOM"
            if status != "WARNING":
                feedback = "Good Depth!"

        elif knee_angle < 140 and self.stage == "STANDING":
            self.stage = "DESCENDING"

        elif knee_angle > 110 and self.stage == "BOTTOM":
            self.stage = "ASCENDING"

        return {
            "knee_angle": int(knee_angle),
            "back_angle": int(back_angle),
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }
