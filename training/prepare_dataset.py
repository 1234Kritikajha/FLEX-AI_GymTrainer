import os
import glob
from PIL import Image
import config

def validate_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def check_dataset():
    print("=" * 60)
    print("YOLO POSE DATASET VERIFICATION SUMMARY")
    print("=" * 60)

    splits = ['train', 'val', 'test']
    total_imgs = 0
    total_lbls = 0

    for split in splits:
        img_dir = os.path.join(config.DATASET_DIR, "images", split)
        lbl_dir = os.path.join(config.DATASET_DIR, "labels", split)

        imgs = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + \
               glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]")) + \
               glob.glob(os.path.join(img_dir, "*.[jJ][pP][eE][gG]"))
        lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))

        img_basenames = {os.path.splitext(os.path.basename(f))[0] for f in imgs}
        lbl_basenames = {os.path.splitext(os.path.basename(f))[0] for f in lbls}

        missing_lbls = img_basenames - lbl_basenames
        orphan_lbls = lbl_basenames - img_basenames

        corrupted = [f for f in imgs if not validate_image(f)]

        print(f"[{split.upper()}]")
        print(f"  Images Found:    {len(imgs)}")
        print(f"  Labels Found:    {len(lbls)}")
        print(f"  Missing Labels:  {len(missing_lbls)}")
        print(f"  Orphan Labels:   {len(orphan_lbls)}")
        print(f"  Corrupted Imgs:  {len(corrupted)}")

        total_imgs += len(imgs)
        total_lbls += len(lbls)

    print("-" * 60)
    print(f"TOTAL IMAGES: {total_imgs} | TOTAL LABELS: {total_lbls}")
    
    if total_imgs == 0:
        print("\nWARNING: Custom YOLO pose image dataset is currently EMPTY.")
        print("Pretrained YOLOv8n-Pose model will be used for keypoint extraction.")
    print("=" * 60)
    return total_imgs, total_lbls

if __name__ == "__main__":
    check_dataset()
