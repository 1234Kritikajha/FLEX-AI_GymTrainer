import os
from ultralytics import YOLO

def train():
    # Base model load from pretrained
    model = YOLO("models/pretrained/yolov8n-pose.pt")

    # Fine-tuning setup
    results = model.train(
        data="dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        project="outputs/training",
        name="flex_pose_run",
        exist_ok=True
    )

    # Move best trained weight to models/trained
    best_weights = "outputs/training/flex_pose_run/weights/best.pt"
    if os.path.exists(best_weights):
        os.system(f"cp {best_weights} models/trained/flex_pose_best.pt")
        print("Trained model saved to models/trained/flex_pose_best.pt")

if __name__ == "__main__":
    train()
