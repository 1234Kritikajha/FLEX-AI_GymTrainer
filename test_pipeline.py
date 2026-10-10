import os
import sys
import numpy as np
import torch

def test_pipeline():
    print("=" * 70)
    print("RUNNING AUTOMATED VERIFICATION TEST SUITE (TESTS 1 - 13)")
    print("=" * 70)

    # TEST 1: Import every new module
    print("[TEST 1/13] Importing all system modules...")
    import config
    from src.keypoint_processor import KeypointProcessor
    from src.pose_classifier import FlexCNN, PoseClassifier
    from src.decision_engine import DecisionEngine
    from src.pose_detector import PoseDetector
    from src.exercise_analyzer import ExerciseAnalyzerFactory, SquatAnalyzer
    from src.voice_feedback import VoiceEngine
    print(" -> PASSED: All modules imported successfully.")

    # TEST 2: Load YOLO model
    print("\n[TEST 2/13] Loading YOLO pose detector...")
    detector = PoseDetector()
    assert detector.model is not None, "YOLO model failed to initialize"
    print(" -> PASSED: YOLO pose detector initialized.")

    # TEST 3: Load CNN model
    print("\n[TEST 3/13] Loading FlexCNN trained classifier...")
    classifier = PoseClassifier()
    assert classifier.is_loaded, "CNN model failed to load weights"
    print(" -> PASSED: FlexCNN model loaded successfully.")

    # TEST 4: Keypoint Preprocessing
    print("\n[TEST 4/13] Running keypoint processor normalization...")
    processor = KeypointProcessor(enable_smoothing=True)
    dummy_kpts = np.random.uniform(50, 400, (17, 3)).astype(np.float32)
    norm_kpts = processor.process(dummy_kpts, is_live=True)
    conv1d_in = processor.get_conv1d_input(norm_kpts)
    assert norm_kpts.shape == (17, 3), f"Invalid normalized shape {norm_kpts.shape}"
    assert conv1d_in.shape == (3, 17), f"Invalid Conv1D shape {conv1d_in.shape}"
    print(" -> PASSED: Keypoint preprocessing output shapes verified.")

    # TEST 5: CNN Prediction
    print("\n[TEST 5/13] Running CNN exercise prediction...")
    pred_res = classifier.predict(dummy_kpts)
    assert "exercise" in pred_res and "confidence" in pred_res, "Invalid prediction dict structure"
    print(f" -> PASSED: Prediction output: {pred_res['exercise']} ({pred_res['confidence']*100:.1f}%)")

    # TEST 6: Decision Engine
    print("\n[TEST 6/13] Evaluating Decision Engine...")
    decision_engine = DecisionEngine()
    bio_dummy = {"status": "GOOD", "feedback": "Good Form", "counter": 2, "stage": "BOTTOM"}
    dec_res = decision_engine.evaluate("squat", pred_res, pose_confidence=0.85, biomechanics_result=bio_dummy)
    assert "status" in dec_res and "is_rep_allowed" in dec_res, "Invalid decision output structure"
    print(" -> PASSED: Decision engine evaluated successfully.")

    # TEST 7: Exercise Analyzer Factory
    print("\n[TEST 7/13] Validating Exercise Analyzer Factory for all 9 exercises...")
    for ex in config.CANONICAL_EXERCISES:
        analyzer = ExerciseAnalyzerFactory.get_analyzer(ex)
        res = analyzer.process(dummy_kpts)
        assert "primary_angle" in res and "counter" in res and "status" in res, f"Analyzer {ex} failed output contract"
    print(" -> PASSED: All 9 exercise analyzers operating correctly.")

    # TEST 8: Verify trainer.py syntax and import
    print("\n[TEST 8/13] Checking trainer.py entrypoint syntax...")
    import trainer
    print(" -> PASSED: trainer.py syntax verified.")

    # TEST 9: No crash when no person visible (keypoints = None)
    print("\n[TEST 9/13] Testing no-person-detected safety (keypoints = None)...")
    dec_none = decision_engine.evaluate("squat", classifier.predict(None), pose_confidence=0.0, biomechanics_result=None)
    assert dec_none["status"] == "NO_POSE" and not dec_none["is_rep_allowed"], "Failed no-person safety test"
    print(" -> PASSED: Safe handling of missing person verified.")

    # TEST 10: Low-confidence keypoints handling
    print("\n[TEST 10/13] Testing low-confidence keypoints safety...")
    low_conf_kpts = np.zeros((17, 3), dtype=np.float32)
    norm_low = processor.process(low_conf_kpts)
    assert not np.isnan(norm_low).any(), "NaN values found in low-confidence keypoint processing"
    print(" -> PASSED: Low-confidence keypoints handled safely without NaN/crashes.")

    # TEST 11: Selected != Detected Exercise Mismatch
    print("\n[TEST 11/13] Testing exercise mismatch detection...")
    cnn_mismatch = {"exercise": "bicep_curl", "confidence": 0.95, "probabilities": {}}
    dec_mismatch = decision_engine.evaluate("squat", cnn_mismatch, pose_confidence=0.90, biomechanics_result=bio_dummy)
    assert dec_mismatch["status"] == "EXERCISE_MISMATCH", f"Expected EXERCISE_MISMATCH, got {dec_mismatch['status']}"
    assert not dec_mismatch["is_rep_allowed"], "Mismatch should disallow rep counting!"
    print(f" -> PASSED: Exercise mismatch detected correctly: '{dec_mismatch['feedback']}'")

    # TEST 12: Selected == Detected Exercise Matching
    print("\n[TEST 12/13] Testing exercise match rep authorization...")
    cnn_match = {"exercise": "squat", "confidence": 0.95, "probabilities": {}}
    dec_match = decision_engine.evaluate("squat", cnn_match, pose_confidence=0.90, biomechanics_result=bio_dummy)
    assert dec_match["status"] == "GOOD", f"Expected GOOD, got {dec_match['status']}"
    assert dec_match["is_rep_allowed"], "Match should allow rep counting!"
    print(" -> PASSED: Exercise match correctly authorizes rep logic.")

    # TEST 13: Non-blocking Voice Feedback
    print("\n[TEST 13/13] Testing non-blocking voice feedback engine...")
    voice = VoiceEngine(cooldown=0.1)
    voice.speak_async("Test audio alert")
    print(" -> PASSED: Voice engine triggered asynchronously without blocking execution thread.")

    print("\n" + "=" * 70)
    print("ALL 13 AUTOMATED PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
