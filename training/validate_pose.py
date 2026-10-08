from ultralytics import YOLO

def validate():
    model_path = "models/trained/flex_pose_best.pt"
    try:
        model = YOLO(model_path)
        metrics = model.val(data="dataset/data.yaml")
        print(f"Validation MAP50-95: {metrics.pose.map}")
    except Exception as e:
        print(f"Validation skip (Train model first): {e}")

if __name__ == "__main__":
    validate()
