import sys
from pathlib import Path
from ultralytics import YOLO

MAP_THRESHOLD = 0.50
model_path = Path("models/yolov8n.pt")
if not model_path.exists():
    sys.exit(1)

model = YOLO(str(model_path))
metrics = model.val(data="coco128.yaml", split="val", verbose=False)
map50 = float(metrics.box.map50)
print(f"mAP@0.5 = {map50:.4f}")

if map50 < MAP_THRESHOLD:
    sys.exit(1)
print("Quality gate aprovado.")
