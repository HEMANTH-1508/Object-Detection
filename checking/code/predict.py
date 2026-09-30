import json
from pathlib import Path
from ultralytics import YOLO

# Current file location
BASE_DIR = Path(__file__).resolve().parent.parent

# Folders inside checking
IMAGES_DIR = BASE_DIR / "images"
OUTPUTS_DIR = BASE_DIR / "outputs"
JSON_DIR = OUTPUTS_DIR / "json"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
MODEL_DIR = BASE_DIR / "model"

# Create output folders
JSON_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Your trained model path
BEST_MODEL = MODEL_DIR / "best.pt"

if not BEST_MODEL.exists():
    raise FileNotFoundError(
        f"Model not found:\n{BEST_MODEL}"
    )

if not IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Images folder not found:\n{IMAGES_DIR}"
    )

print("Loading model...")
model = YOLO(str(BEST_MODEL))

print("Starting prediction...")

results = model.predict(
    source=str(IMAGES_DIR),
    conf=0.25,
    imgsz=640,
    save=True,
    save_txt=False,
    show=False,
    project=str(ANNOTATED_DIR),
    name="predictions",
    exist_ok=True,
)

print("Creating JSON files...")

for result in results:

    image_path = Path(result.path)

    output = {
        "image": image_path.name,
        "detections": []
    }

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(boxes.cls[i].item())
            class_name = result.names[class_id]
            confidence = float(boxes.conf[i].item())

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            output["detections"].append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

    json_file = JSON_DIR / f"{image_path.stem}.json"

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4
        )

    print(f"Created: {json_file}")

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)
print(f"Annotated Images: {ANNOTATED_DIR}")
print(f"JSON Files: {JSON_DIR}")