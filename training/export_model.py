from ultralytics import YOLO

def export():
    try:
        model = YOLO("models/trained/flex_pose_best.pt")
        model.export(format="onnx")
        print("Model exported to ONNX successfully.")
    except Exception as e:
        print(f"Export error: {e}")

if __name__ == "__main__":
    export()
