import os

def check_dataset():
    splits = ['train', 'val', 'test']
    for split in splits:
        img_path = f"dataset/images/{split}"
        lbl_path = f"dataset/labels/{split}"
        
        imgs = os.listdir(img_path) if os.path.exists(img_path) else []
        lbls = os.listdir(lbl_path) if os.path.exists(lbl_path) else []
        
        print(f"[{split.upper()}] Images: {len(imgs)} | Labels: {len(lbls)}")

if __name__ == "__main__":
    check_dataset()
