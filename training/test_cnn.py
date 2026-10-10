import numpy as np
import config
from src.pose_classifier import PoseClassifier

def test_cnn_sample():
    print("=" * 60)
    print("TESTING FLEX-CNN INFERENCE PIPELINE")
    print("=" * 60)

    classifier = PoseClassifier()

    if not classifier.is_loaded:
        print("Warning: CNN trained weights not loaded. Run train_cnn.py first.")
        return

    # Create a synthetic squat keypoint sample
    sample_keypoints = np.array([
        [300, 100, 0.9], [295, 95, 0.9], [305, 95, 0.9], [290, 95, 0.8], [310, 95, 0.8],
        [270, 160, 0.9], [330, 160, 0.9], [250, 230, 0.85], [350, 230, 0.85],
        [250, 300, 0.8], [350, 300, 0.8], [280, 370, 0.9], [320, 370, 0.9], # Hips dropped
        [270, 470, 0.9], [330, 470, 0.9], [280, 570, 0.85], [320, 570, 0.85]
    ], dtype=np.float32)

    result = classifier.predict(sample_keypoints)

    display_name = config.EXERCISE_DISPLAY_NAMES.get(result['exercise'], result['exercise'])
    print(f"Predicted exercise: {display_name} ({result['exercise']})")
    print(f"Confidence:         {result['confidence'] * 100:.1f}%")
    print("\nClass Probabilities Breakdown:")
    for ex, prob in result['probabilities'].items():
        print(f"  {ex:<20}: {prob * 100:5.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    test_cnn_sample()
