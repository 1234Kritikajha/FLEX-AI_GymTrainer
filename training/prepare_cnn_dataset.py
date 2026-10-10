import os
import json
import torch
import numpy as np
import config
from src.keypoint_processor import KeypointProcessor

def generate_synthetic_exercise_keypoints(exercise, num_sequences=30, frames_per_seq=20):
    """
    Generates realistic biomechanical 17-keypoint sequences for training and validating FlexCNN.
    COCO 17 Keypoints:
    0: nose, 1: L_eye, 2: R_eye, 3: L_ear, 4: R_ear,
    5: L_shoulder, 6: R_shoulder, 7: L_elbow, 8: R_elbow, 9: L_wrist, 10: R_wrist,
    11: L_hip, 12: R_hip, 13: L_knee, 14: R_knee, 15: L_ankle, 16: R_ankle
    """
    sequences = []
    
    for seq in range(num_sequences):
        # Vary body height and perspective scale
        scale = np.random.uniform(0.85, 1.15)
        offset_x = np.random.uniform(200, 400)
        offset_y = np.random.uniform(150, 300)
        
        # Base standing coordinates
        base_kpts = np.array([
            [300, 100], [295, 95], [305, 95], [290, 95], [310, 95], # Head
            [270, 160], [330, 160], # Shoulders
            [250, 230], [350, 230], # Elbows
            [250, 300], [350, 300], # Wrists
            [280, 300], [320, 300], # Hips
            [280, 420], [320, 420], # Knees
            [280, 540], [320, 540]  # Ankles
        ], dtype=np.float32)
        
        # Apply random body variation
        base_kpts = base_kpts * scale + np.array([offset_x - 300, offset_y - 100])
        
        for f in range(frames_per_seq):
            t = (f / float(frames_per_seq)) * 2 * np.pi
            kpts = base_kpts.copy()
            
            # Exercise specific kinematics
            if exercise == "squat":
                drop = np.sin(t) * 70.0 if np.sin(t) > 0 else 0.0
                kpts[11:, 1] += drop # Hips drop
                kpts[13:, 1] += drop * 0.7 # Knees drop
                kpts[13, 0] -= drop * 0.3 # Knee bend outward
                kpts[14, 0] += drop * 0.3
                kpts[5:11, 1] += drop # Upper body follows hips
            elif exercise == "bicep_curl":
                flex = (np.sin(t) + 1.0) / 2.0 # 0.0 to 1.0
                kpts[9, 1] = kpts[7, 1] - flex * 70.0 # Left wrist curls up
                kpts[10, 1] = kpts[8, 1] - flex * 70.0 # Right wrist curls up
            elif exercise == "pushup":
                # Horizontal orientation
                kpts[:, 1] = base_kpts[:, 0] * 0.6 + offset_y
                kpts[:, 0] = np.linspace(150, 500, 17)
                dip = (np.sin(t) + 1.0) * 25.0
                kpts[5:11, 1] += dip # Chest/shoulders lower
            elif exercise == "lunge":
                drop = (np.sin(t) + 1.0) * 35.0
                kpts[11:, 1] += drop # Hips lower
                kpts[13, 0] += drop * 0.4 # Front leg steps forward
            elif exercise == "plank":
                # Horizontal straight line with minor micro-vibration
                kpts[:, 1] = 350.0 + np.random.normal(0, 1.5, 17)
                kpts[:, 0] = np.linspace(150, 500, 17)
            elif exercise == "glute_bridge":
                # Supine position
                kpts[:, 1] = 400.0
                kpts[:, 0] = np.linspace(150, 500, 17)
                lift = (np.sin(t) + 1.0) * 45.0
                kpts[11:13, 1] -= lift # Hips thrust upward
            elif exercise == "crunch":
                kpts[:, 1] = 400.0
                kpts[:, 0] = np.linspace(150, 500, 17)
                curl = (np.sin(t) + 1.0) * 30.0
                kpts[0:7, 1] -= curl # Head/shoulders flex up
            elif exercise == "shoulder_press":
                lift = (np.sin(t) + 1.0) * 60.0
                kpts[7:11, 1] -= lift # Hands reach overhead
            elif exercise == "tricep_extension":
                ext = (np.sin(t) + 1.0) * 50.0
                kpts[9:11, 1] -= ext # Forearms extend overhead
                
            # Add realistic sensor noise and joint confidence
            noise = np.random.normal(0, 2.0, kpts.shape).astype(np.float32)
            kpts += noise
            conf = np.random.uniform(0.75, 0.98, (17, 1)).astype(np.float32)
            
            kpts_with_conf = np.hstack([kpts, conf])
            sequences.append((kpts_with_conf, exercise))
            
    return sequences

def prepare_cnn_dataset():
    print("=" * 60)
    print("PREPARING CNN FEATURE DATASET FOR EXERCISE RECOGNITION")
    print("=" * 60)
    
    processor = KeypointProcessor(enable_smoothing=False)
    exercises = config.CANONICAL_EXERCISES
    class_to_idx = {ex: i for i, ex in enumerate(exercises)}
    idx_to_class = {i: ex for i, ex in enumerate(exercises)}
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    summary = {}
    
    for ex in exercises:
        # Generate clean sequences for each exercise
        seqs = generate_synthetic_exercise_keypoints(ex, num_sequences=40, frames_per_seq=25)
        
        # Split sequences to prevent temporal data leakage (70% train, 15% val, 15% test)
        n_total = len(seqs)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        train_samples = seqs[:n_train]
        val_samples = seqs[n_train:n_train + n_val]
        test_samples = seqs[n_train + n_val:]
        
        for kpts, _ in train_samples:
            norm_kpts = processor.process(kpts)
            conv1d_feat = processor.get_conv1d_input(norm_kpts) # (3, 17)
            X_train.append(conv1d_feat)
            y_train.append(class_to_idx[ex])
            
        for kpts, _ in val_samples:
            norm_kpts = processor.process(kpts)
            conv1d_feat = processor.get_conv1d_input(norm_kpts)
            X_val.append(conv1d_feat)
            y_val.append(class_to_idx[ex])
            
        for kpts, _ in test_samples:
            norm_kpts = processor.process(kpts)
            conv1d_feat = processor.get_conv1d_input(norm_kpts)
            X_test.append(conv1d_feat)
            y_test.append(class_to_idx[ex])
            
        summary[ex] = {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        }
        
    out_dir = os.path.join(config.DATASET_DIR, "pose_classifier")
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save({"X": torch.tensor(np.array(X_train), dtype=torch.float32), "y": torch.tensor(y_train, dtype=torch.long)}, os.path.join(out_dir, "train.pt"))
    torch.save({"X": torch.tensor(np.array(X_val), dtype=torch.float32), "y": torch.tensor(y_val, dtype=torch.long)}, os.path.join(out_dir, "val.pt"))
    torch.save({"X": torch.tensor(np.array(X_test), dtype=torch.float32), "y": torch.tensor(y_test, dtype=torch.long)}, os.path.join(out_dir, "test.pt"))
    
    class_mapping = {"class_to_idx": class_to_idx, "idx_to_class": {str(k): v for k, v in idx_to_class.items()}}
    with open(os.path.join(config.TRAINED_MODELS_DIR, "exercise_classes.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
        
    print("\nDataset Summary")
    print(f"{'Exercise/Class':<20} {'Train':<10} {'Validation':<12} {'Test':<10}")
    print("-" * 55)
    for ex, counts in summary.items():
        print(f"{ex:<20} {counts['train']:<10} {counts['val']:<12} {counts['test']:<10}")
    print("=" * 60)
    print(f"CNN Dataset created successfully in '{out_dir}'")
    print(f"Total samples: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_cnn_dataset()
