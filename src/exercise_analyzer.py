from src.angle_utils import calculate_angle

class SquatAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "STANDING"

    def process(self, kpts):
        shoulder, hip, knee, ankle = kpts[5], kpts[11], kpts[13], kpts[15]

        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)

        feedback = "Ready"
        status = "GOOD"

        if back_angle < 120:
            feedback = "Keep Back Straight!"
            status = "WARNING"

        # Squat State Machine with smoothed thresholds
        if knee_angle > 150:
            if self.stage == "BOTTOM" or self.stage == "ASCENDING":
                self.counter += 1
                feedback = f"Squat Rep {self.counter} Complete!"
            self.stage = "STANDING"
        elif knee_angle < 115:
            self.stage = "BOTTOM"
            if status != "WARNING":
                feedback = "Good Depth!"
        elif knee_angle < 145 and self.stage == "STANDING":
            self.stage = "DESCENDING"

        return {
            "primary_angle": int(knee_angle),
            "angle_name": "Knee Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }

class BicepCurlAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "DOWN"

    def process(self, kpts):
        shoulder, elbow, wrist = kpts[5], kpts[7], kpts[9]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        feedback = "Good Form"
        status = "GOOD"

        # Bicep Curl State Machine
        if elbow_angle > 140:
            if self.stage == "UP":
                self.counter += 1
                feedback = f"Bicep Rep {self.counter} Complete!"
            self.stage = "DOWN"
        elif elbow_angle < 50:
            self.stage = "UP"
            feedback = "Curl Complete, Lower Slow"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Elbow Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }

class PushupAnalyzer:
    def __init__(self):
        self.counter = 0
        self.stage = "UP"

    def process(self, kpts):
        shoulder, elbow, wrist = kpts[5], kpts[7], kpts[9]
        hip, ankle = kpts[11], kpts[15]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, ankle)

        feedback = "Push Up"
        status = "GOOD"

        if body_angle < 140:
            feedback = "Keep Body Straight!"
            status = "WARNING"

        if elbow_angle > 150:
            if self.stage == "DOWN":
                self.counter += 1
                feedback = f"Pushup Rep {self.counter} Complete!"
            self.stage = "UP"
        elif elbow_angle < 100:
            self.stage = "DOWN"
            if status != "WARNING":
                feedback = "Good Depth!"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Elbow Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }
