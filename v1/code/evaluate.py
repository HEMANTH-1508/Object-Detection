from ultralytics import YOLO
from config import (
    DATA_YAML,
    MODEL_DIR,
    EVALUATION_DIR,
)

BEST_MODEL = (
    MODEL_DIR
    / "monitor_detector"
    / "weights"
    / "best.pt"
)

if not BEST_MODEL.exists():

    raise FileNotFoundError(
        f"Model not found:\n{BEST_MODEL}\n\n"
        "Train the model first using train.py."
    )

print("Loading trained model...")
model = YOLO(str(BEST_MODEL))


print("Starting evaluation...")
metrics = model.val(
    data=str(DATA_YAML),
    imgsz=640,
    project=str(EVALUATION_DIR),
    name="monitor_evaluation",
    exist_ok=True,
    plots=True,
)

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


print()
print(
    f"mAP50: "
    f"{metrics.box.map50:.4f}"
)

print(
    f"mAP50-95: "
    f"{metrics.box.map:.4f}"
)

print(
    f"Precision: "
    f"{metrics.box.mp:.4f}"
)

print(
    f"Recall: "
    f"{metrics.box.mr:.4f}"
)


print()
print(
    f"Evaluation results saved to:"
)

print(EVALUATION_DIR)