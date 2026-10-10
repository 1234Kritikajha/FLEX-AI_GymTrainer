import sys
from training.prepare_dataset import check_dataset
from training.train_pose import train_pose
from training.validate_pose import validate
from training.prepare_cnn_dataset import prepare_cnn_dataset
from training.train_cnn import train_cnn
from training.validate_cnn import validate_cnn
from training.test_cnn import test_cnn_sample

def run_pipeline():
    print("#" * 70)
    print("STARTING COMPLETE FLEX AI TRAINER RETRAINING & VALIDATION PIPELINE")
    print("#" * 70)
    
    # 1. Check YOLO Dataset
    print("\n--- STEP 1: Inspecting YOLO Pose Dataset ---")
    check_dataset()
    
    # 2. Train YOLO Pose
    print("\n--- STEP 2: Fine-Tuning YOLOv8-Pose Model ---")
    train_pose(epochs=10)
    
    # 3. Validate YOLO Pose
    print("\n--- STEP 3: Validating YOLOv8-Pose Model ---")
    validate()
    
    # 4. Prepare CNN Dataset
    print("\n--- STEP 4: Preparing Keypoint Feature Dataset for CNN ---")
    prepare_cnn_dataset()
    
    # 5. Train CNN Model
    print("\n--- STEP 5: Training FlexCNN Classifier Model ---")
    train_cnn(epochs=35)
    
    # 6. Validate CNN Model
    print("\n--- STEP 6: Evaluating CNN Model Performance ---")
    validate_cnn()
    
    # 7. Test CNN Single Inference
    print("\n--- STEP 7: Running Single Sample Inference Test ---")
    test_cnn_sample()
    
    print("\n" + "#" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("#" * 70)

if __name__ == "__main__":
    run_pipeline()
