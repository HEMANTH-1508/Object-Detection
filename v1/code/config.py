from pathlib import Path

V1_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = V1_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_DIR = V1_DIR / "Model"
OUTPUT_DIR = V1_DIR / "Output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
JSON_DIR = OUTPUT_DIR / "json"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

BASE_MODEL = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)