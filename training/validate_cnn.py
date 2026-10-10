import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import config
from src.pose_classifier import FlexCNN

def validate_cnn():
    print("=" * 60)
    print("EVALUATING CNN EXERCISE CLASSIFIER MODEL PERFORMANCE")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = os.path.join(config.DATASET_DIR, "pose_classifier")
    test_file = os.path.join(data_dir, "test.pt")
    val_file = os.path.join(data_dir, "val.pt")

    if not os.path.exists(test_file) or not os.path.exists(config.CNN_TRAINED_PATH):
        raise FileNotFoundError("Trained CNN model or test dataset missing. Train model first!")

    class_mapping_path = os.path.join(config.TRAINED_MODELS_DIR, "exercise_classes.json")
    idx_to_class = {i: ex for i, ex in enumerate(config.CANONICAL_EXERCISES)}

    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)
            idx_to_class = {int(k): v for k, v in mapping.get("idx_to_class", {}).items()}

    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    # Load model
    model = FlexCNN(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(config.CNN_TRAINED_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    test_data = torch.load(test_file)
    test_loader = DataLoader(TensorDataset(test_data["X"], test_data["y"]), batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_b.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = (all_preds == all_targets).mean() * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    print(f"Overall Test Accuracy: {acc:.2f}%")
    print(f"Weighted Precision:   {precision:.4f}")
    print(f"Weighted Recall:      {recall:.4f}")
    print(f"Weighted F1 Score:    {f1:.4f}\n")

    print("Per-Class Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    header = f"{'Class':<18}" + "".join([f"{name[:6]:>8}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<18}" + "".join([f"{val:>8}" for val in row])
        print(row_str)

    output_dir = os.path.join(config.OUTPUTS_DIR, "training", "cnn")
    os.makedirs(output_dir, exist_ok=True)
    report_dict = {
        "test_accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(output_dir, "validation_metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    print("=" * 60)
    return acc

if __name__ == "__main__":
    validate_cnn()
