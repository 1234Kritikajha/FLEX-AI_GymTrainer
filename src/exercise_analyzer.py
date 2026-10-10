import time
import numpy as np
from src.angle_utils import calculate_angle
import config

def get_side_kpts(kpts, left_idx, right_idx):
    """
    Selects the side (left or right) with higher confidence/visibility.
    Returns (pt1, pt2, pt3, side_name)
    """
    kpts_arr = np.array(kpts)
    has_conf = kpts_arr.shape[1] >= 3

    l_conf = sum(kpts_arr[i, 2] for i in left_idx) if has_conf else 3.0
    r_conf = sum(kpts_arr[i, 2] for i in right_idx) if has_conf else 3.0

    if r_conf > l_conf:
        return [kpts_arr[i, :2] for i in right_idx], "RIGHT"
    return [kpts_arr[i, :2] for i in left_idx], "LEFT"


class BaseAnalyzer:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.stage = "START"

    def reset(self):
        self.counter = 0
        self.stage = "START"


class SquatAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("squat")
        self.stage = "STANDING"

    def process(self, kpts):
        pts, side = get_side_kpts(kpts, [5, 11, 13], [6, 12, 14])  # shoulder, hip, knee
        sh, hip, knee = pts
        pts_leg, _ = get_side_kpts(kpts, [11, 13, 15], [12, 14, 16])  # hip, knee, ankle
        _, _, ankle = pts_leg

        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(sh, hip, knee)

        feedback = "Good Squat Form"
        status = "GOOD"

        if back_angle < 120:
            feedback = "Keep Back Straight!"
            status = "WARNING"

        if knee_angle > 155:
            if self.stage == "BOTTOM" or self.stage == "ASCENDING":
                self.counter += 1
                feedback = f"Squat Rep {self.counter} Complete!"
            self.stage = "STANDING"
        elif knee_angle < 105:
            self.stage = "BOTTOM"
            if status != "WARNING":
                feedback = "Good Squat Depth!"
        elif knee_angle < 145 and self.stage == "STANDING":
            self.stage = "DESCENDING"
        elif knee_angle > 115 and self.stage == "BOTTOM":
            self.stage = "ASCENDING"

        return {
            "primary_angle": int(knee_angle),
            "angle_name": "Knee Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class BicepCurlAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("bicep_curl")
        self.stage = "DOWN"

    def process(self, kpts):
        pts, side = get_side_kpts(kpts, [5, 7, 9], [6, 8, 10])  # shoulder, elbow, wrist
        sh, elbow, wrist = pts

        elbow_angle = calculate_angle(sh, elbow, wrist)

        feedback = "Good Curl Form"
        status = "GOOD"

        if elbow_angle > 145:
            if self.stage == "UP":
                self.counter += 1
                feedback = f"Bicep Rep {self.counter} Complete!"
            self.stage = "DOWN"
        elif elbow_angle < 50:
            self.stage = "UP"
            feedback = "Curl Peak! Lower Slow"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Elbow Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class PushupAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("pushup")
        self.stage = "UP"

    def process(self, kpts):
        arm_pts, _ = get_side_kpts(kpts, [5, 7, 9], [6, 8, 10])
        sh, elbow, wrist = arm_pts
        body_pts, _ = get_side_kpts(kpts, [5, 11, 15], [6, 12, 16])
        b_sh, hip, ankle = body_pts

        elbow_angle = calculate_angle(sh, elbow, wrist)
        body_angle = calculate_angle(b_sh, hip, ankle)

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
                feedback = "Good Pushup Depth!"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Elbow Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class LungeAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("lunge")
        self.stage = "STANDING"

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [11, 13, 15], [12, 14, 16])  # hip, knee, ankle
        hip, knee, ankle = pts

        knee_angle = calculate_angle(hip, knee, ankle)

        feedback = "Good Lunge Form"
        status = "GOOD"

        if knee_angle > 155:
            if self.stage == "BOTTOM" or self.stage == "UP":
                self.counter += 1
                feedback = f"Lunge Rep {self.counter} Complete!"
            self.stage = "STANDING"
        elif knee_angle < 100:
            self.stage = "BOTTOM"
            feedback = "Good Lunge Depth!"
        elif knee_angle < 145 and self.stage == "STANDING":
            self.stage = "DOWN"

        return {
            "primary_angle": int(knee_angle),
            "angle_name": "Front Knee Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class PlankAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("plank")
        self.stage = "PLANK_HOLD"
        self.start_time = None
        self.hold_duration = 0

    def reset(self):
        super().reset()
        self.start_time = None
        self.hold_duration = 0

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [5, 11, 15], [6, 12, 16])  # shoulder, hip, ankle
        sh, hip, ankle = pts

        body_angle = calculate_angle(sh, hip, ankle)

        feedback = "Hold Plank"
        status = "GOOD"

        if body_angle < 140:
            feedback = "Hips Dropping! Lift Hips"
            status = "WARNING"
            self.start_time = None
        elif body_angle > 175:
            feedback = "Hips Too High! Lower Hips"
            status = "WARNING"
            self.start_time = None
        else:
            if self.start_time is None:
                self.start_time = time.time()
            self.hold_duration = int(time.time() - self.start_time)
            self.counter = self.hold_duration
            feedback = f"Holding Plank: {self.hold_duration}s"

        return {
            "primary_angle": int(body_angle),
            "angle_name": "Body Line Angle",
            "counter": self.counter,
            "stage": "PLANK_HOLD",
            "feedback": feedback,
            "status": status
        }


class GluteBridgeAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("glute_bridge")
        self.stage = "DOWN"

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [5, 11, 13], [6, 12, 14])  # shoulder, hip, knee
        sh, hip, knee = pts

        hip_angle = calculate_angle(sh, hip, knee)

        feedback = "Glute Bridge"
        status = "GOOD"

        if hip_angle > 155:
            if self.stage == "DOWN":
                self.counter += 1
                feedback = f"Bridge Rep {self.counter} Complete!"
            self.stage = "UP"
        elif hip_angle < 120:
            self.stage = "DOWN"
            feedback = "Squeeze Glutes at Top"

        return {
            "primary_angle": int(hip_angle),
            "angle_name": "Hip Extension",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class CrunchAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("crunch")
        self.stage = "DOWN"

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [5, 11, 13], [6, 12, 14])  # shoulder, hip, knee
        sh, hip, knee = pts

        torso_angle = calculate_angle(sh, hip, knee)

        feedback = "Ab Crunch"
        status = "GOOD"

        if torso_angle < 110:
            if self.stage == "DOWN":
                self.counter += 1
                feedback = f"Crunch Rep {self.counter} Complete!"
            self.stage = "UP"
        elif torso_angle > 140:
            self.stage = "DOWN"
            feedback = "Engage Core, Flex Up"

        return {
            "primary_angle": int(torso_angle),
            "angle_name": "Torso Flex Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class ShoulderPressAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("shoulder_press")
        self.stage = "DOWN"

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [11, 5, 7], [12, 6, 8])  # hip, shoulder, elbow
        hip, sh, elbow = pts
        arm_pts, _ = get_side_kpts(kpts, [5, 7, 9], [6, 8, 10])  # shoulder, elbow, wrist
        _, _, wrist = arm_pts

        elbow_angle = calculate_angle(sh, elbow, wrist)

        feedback = "Shoulder Press"
        status = "GOOD"

        if elbow_angle > 150:
            if self.stage == "DOWN":
                self.counter += 1
                feedback = f"Press Rep {self.counter} Complete!"
            self.stage = "UP"
        elif elbow_angle < 90:
            self.stage = "DOWN"
            feedback = "Press Overhead"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Elbow Lockout",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class TricepExtensionAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("tricep_extension")
        self.stage = "BENT"

    def process(self, kpts):
        pts, _ = get_side_kpts(kpts, [5, 7, 9], [6, 8, 10])  # shoulder, elbow, wrist
        sh, elbow, wrist = pts

        elbow_angle = calculate_angle(sh, elbow, wrist)

        feedback = "Tricep Extension"
        status = "GOOD"

        if elbow_angle > 150:
            if self.stage == "BENT":
                self.counter += 1
                feedback = f"Tricep Rep {self.counter} Complete!"
            self.stage = "EXTENDED"
        elif elbow_angle < 85:
            self.stage = "BENT"
            feedback = "Extend Arm Fully"

        return {
            "primary_angle": int(elbow_angle),
            "angle_name": "Tricep Extension Angle",
            "counter": self.counter,
            "stage": self.stage,
            "feedback": feedback,
            "status": status
        }


class ExerciseAnalyzerFactory:
    """
    Factory to create and retrieve modular exercise analyzers by canonical name.
    """
    _analyzers = {
        "squat": SquatAnalyzer,
        "bicep_curl": BicepCurlAnalyzer,
        "pushup": PushupAnalyzer,
        "lunge": LungeAnalyzer,
        "plank": PlankAnalyzer,
        "glute_bridge": GluteBridgeAnalyzer,
        "crunch": CrunchAnalyzer,
        "shoulder_press": ShoulderPressAnalyzer,
        "tricep_extension": TricepExtensionAnalyzer
    }

    @classmethod
    def get_analyzer(cls, canonical_name):
        name = canonical_name.lower().strip()
        if name in cls._analyzers:
            return cls._analyzers[name]()
        raise ValueError(f"Unknown exercise name: '{canonical_name}'. Supported: {list(cls._analyzers.keys())}")
