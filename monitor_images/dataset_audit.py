import json
import yaml
from pathlib import Path
from PIL import Image
import cv2

DATASET_ROOT = Path(".")
OUTPUT_DIR = Path("../outputs/monitor_images")
BLUR_THRESHOLD = 100
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "total_images": 0,
    "corrupted_count": 0,
    "grayscale_count": 0,
    "blurry_count": 0,
    "corrupted": [],
    "grayscale": [],
    "blurry": [],
    "resolutions": [],
    "statistics": {}
}


for split in ["train", "val", "test"]:
    split_dir = DATASET_ROOT / split
    if not split_dir.exists():
        print(f"Missing folder: {split_dir}")
        continue

    print(f"Scanning: {split_dir}")
    for img_path in split_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            img = Image.open(img_path)
            report["total_images"] += 1
            width, height = img.size
            report["resolutions"].append({
                "file": str(img_path),
                "width": width,
                "height": height
            })

            if img.mode == "L":
                report["grayscale"].append(str(img_path))
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                raise Exception("OpenCV could not read image")
            gray = cv2.cvtColor(
                cv_img,
                cv2.COLOR_BGR2GRAY
            )

            blur_score = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            if blur_score < BLUR_THRESHOLD:
                report["blurry"].append({
                    "file": str(img_path),
                    "blur_score": float(blur_score)
                })

        except Exception as e:
            report["corrupted"].append({
                "file": str(img_path),
                "error": str(e)
            })

widths = [
    item["width"]
    for item in report["resolutions"]
]

heights = [
    item["height"]
    for item in report["resolutions"]
]

report["corrupted_count"] = len(
    report["corrupted"]
)

report["grayscale_count"] = len(
    report["grayscale"]
)

report["blurry_count"] = len(
    report["blurry"]
)

if widths and heights:

    report["statistics"] = {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": round(sum(widths) / len(widths), 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": round(sum(heights) / len(heights), 2)
    }

json_path = OUTPUT_DIR / "dataset_report.json"
yaml_path = OUTPUT_DIR / "dataset_report.yaml"

with open(json_path, "w") as f:
    json.dump(report, f, indent=4)

with open(yaml_path, "w") as f:
    yaml.dump(report, f, sort_keys=False)

print("\n=================================")
print("DATASET AUDIT COMPLETE")
print("=================================")
print(f"Total Images     : {report['total_images']}")
print(f"Corrupted Images : {report['corrupted_count']}")
print(f"Grayscale Images : {report['grayscale_count']}")
print(f"Blurry Images    : {report['blurry_count']}")
print(f"JSON Report      : {json_path}")
print(f"YAML Report      : {yaml_path}")
print("=================================")