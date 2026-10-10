import os
import shutil
import torch
from ultralytics import YOLO
import config
from training.prepare_dataset import check_dataset

def train_pose(epochs=30, imgsz=640, batch=16):
    print("=" * 60)
    print("TRAINING / FINE-TUNING YOLOv8-POSE MODEL")
    print("=" * 60)

    total_imgs, total_lbls = check_dataset()

    if total_imgs == 0:
        print("\nNotice: Custom image dataset is empty.")
        print(f"Preserving pretrained base model at '{config.YOLO_PRETRAINED_PATH}'.")
        os.makedirs(config.TRAINED_MODELS_DIR, exist_ok=True)
        if os.path.exists(config.YOLO_PRETRAINED_PATH):
            shutil.copy2(config.YOLO_PRETRAINED_PATH, config.YOLO_TRAINED_PATH)
            print(f"Copied base weights to '{config.YOLO_TRAINED_PATH}' as initial flex_pose_best.pt.")
        print("Add labeled YOLO pose images to 'dataset/images/train' to enable custom pose fine-tuning.")
        print("=" * 60)
        return

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"YOLO Training Hardware Device: {device}")

    model = YOLO(config.YOLO_PRETRAINED_PATH)

    output_dir = os.path.join(config.OUTPUTS_DIR, "training", "yolo")
    results = model.train(
        data=os.path.join(config.DATASET_DIR, "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=output_dir,
        name="flex_pose_run",
        exist_ok=True
    )

    best_weights = os.path.join(output_dir, "flex_pose_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, config.YOLO_TRAINED_PATH)
        print(f"Trained YOLO pose model saved to '{config.YOLO_TRAINED_PATH}'")

    print("=" * 60)

if __name__ == "__main__":
    train_pose()
