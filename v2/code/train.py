from ultralytics import YOLO
from config import (
    DATA_YAML,
    BASE_MODEL,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_DIR,
)



print("Loading YOLO model...")
model = YOLO(BASE_MODEL)

print("Starting training...")
results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    project=str(MODEL_DIR),
    name="monitor_detector",

    save=True,
    plots=True,
    pretrained=True,
    verbose=True,
)

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print()
print("Training results:")
print(results)

print()
print(f"Model directory: {MODEL_DIR}")