import json
from pathlib import Path
from ultralytics import YOLO
from config import (
    PROJECT_DIR,
    MODEL_DIR,
    ANNOTATED_DIR,
    JSON_DIR,
)

TEST_IMAGES_DIR = PROJECT_DIR / "dataset_v2" / "original_data" / "test" / "images"

BEST_MODEL = (
    MODEL_DIR
    / "monitor_detector"
    / "weights"
    / "best.pt"
)

if not BEST_MODEL.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{BEST_MODEL}\n\n"
        "Run train.py first."
    )

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Test images directory not found:\n{TEST_IMAGES_DIR}"
    )


print("Loading trained model...")
model = YOLO(str(BEST_MODEL))


print("Starting prediction...")

results = model.predict(
    source=str(TEST_IMAGES_DIR),
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
    image_name = image_path.name

    output = {
        "image": image_name,
        "detections": []
    }

    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i].item()
            )

            class_name = result.names[class_id]

            confidence = float(
                boxes.conf[i].item()
            )

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence,
                    4
                ),

                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            }

            output["detections"].append(
                detection
            )

    json_filename = image_path.stem + ".json"
    json_path = JSON_DIR / json_filename

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=4
        )
    print(f"Created: {json_path}")

print()
print("=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print(f"Annotated images: {ANNOTATED_DIR}")
print(f"JSON files:       {JSON_DIR}")