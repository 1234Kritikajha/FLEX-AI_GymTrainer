import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import config
from src.pose_classifier import FlexCNN

def train_cnn(epochs=35, batch_size=32, lr=0.001):
    print("=" * 60)
    print("TRAINING FLEX-CNN EXERCISE CLASSIFIER MODEL")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Hardware Device: {device}")
    
    data_dir = os.path.join(config.DATASET_DIR, "pose_classifier")
    train_file = os.path.join(data_dir, "train.pt")
    val_file = os.path.join(data_dir, "val.pt")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError("Processed dataset files missing! Run python training/prepare_cnn_dataset.py first.")
        
    train_data = torch.load(train_file)
    val_data = torch.load(val_file)
    
    train_dataset = TensorDataset(train_data["X"], train_data["y"])
    val_dataset = TensorDataset(val_data["X"], val_data["y"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    class_mapping_path = os.path.join(config.TRAINED_MODELS_DIR, "exercise_classes.json")
    num_classes = len(config.CANONICAL_EXERCISES)
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)
            num_classes = len(mapping.get("class_to_idx", {}))
            
    model = FlexCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_b.size(0)
            _, predicted = outputs.max(1)
            total += y_b.size(0)
            correct += predicted.eq(y_b).sum().item()
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0
        
        # Validation phase
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_running_loss += loss.item() * X_b.size(0)
                _, predicted = outputs.max(1)
                val_total += y_b.size(0)
                val_correct += predicted.eq(y_b).sum().item()
                
        val_loss = val_running_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        
        scheduler.step(val_acc)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes
            }
            torch.save(checkpoint, config.CNN_TRAINED_PATH)
            
    # Save training outputs and curves
    output_cnn_dir = os.path.join(config.OUTPUTS_DIR, "training", "cnn")
    os.makedirs(output_cnn_dir, exist_ok=True)
    
    with open(os.path.join(output_cnn_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_cnn_dir, "learning_curves.png"))
    plt.close()
    
    print("=" * 60)
    print(f"Training Finished! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.CNN_TRAINED_PATH}")
    print(f"Curves saved to: {os.path.join(output_cnn_dir, 'learning_curves.png')}")
    print("=" * 60)
    return best_val_acc

if __name__ == "__main__":
    train_cnn()
