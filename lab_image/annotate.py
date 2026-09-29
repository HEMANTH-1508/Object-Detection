import json
import joblib
import torch
import timm
import numpy as np
import cv2

from pathlib import Path
from PIL import Image
import torchvision.transforms as T


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path(
    "../models/monitor_images/nearest_neighbor_v2.pkl"
)

THRESHOLD_PATH = Path(
    "../models/monitor_images/threshold_v2.json"
)

INPUT_DIR = Path("main/test")

OUTPUT_DIR = Path(
    "../outputs/monitor_images/annotated"
)

OUTPUT_PREFIX = "monitor"

IMAGE_SIZE = 224
N_NEIGHBORS = 5

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = T.Compose([
    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

print("Loading EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# LOAD NEAREST NEIGHBOR MODEL
# ============================================================

print("Loading nearest-neighbor model...")

nn_model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

print("Loading threshold...")

with open(
    THRESHOLD_PATH,
    "r"
) as f:

    threshold_data = json.load(f)


THRESHOLD = float(
    threshold_data["threshold"]
)

print(
    f"Threshold: {THRESHOLD:.6f}"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    pil_image = Image.fromarray(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    tensor = transform(
        pil_image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        features = model(
            tensor
        )

    features = (
        features
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Same normalization used during training
    features = features / (
        np.linalg.norm(
            features,
            axis=1,
            keepdims=True
        ) + 1e-10
    )

    return features


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    features = extract_features(
        image
    )

    distances, _ = nn_model.kneighbors(
        features,
        n_neighbors=min(
            N_NEIGHBORS,
            nn_model.n_samples_fit_
        )
    )

    anomaly_score = float(
        np.mean(distances)
    )

    if anomaly_score > THRESHOLD:

        label = "ANOMALY"

    else:

        label = "MONITOR"

    score_ratio = (
        anomaly_score /
        THRESHOLD
    )

    return {
        "label": label,
        "anomaly_score": anomaly_score,
        "score_ratio": score_ratio
    }


# ============================================================
# DRAW ANNOTATION
# ============================================================

def annotate_image(
    image,
    result
):

    height, width = image.shape[:2]

    label = result["label"]
    score = result["anomaly_score"]

    # Whole-image bounding box
    x1 = 5
    y1 = 5
    x2 = width - 5
    y2 = height - 5

    # OpenCV BGR
    if label == "MONITOR":

        box_color = (
            0,
            255,
            0
        )

    else:

        box_color = (
            0,
            0,
            255
        )

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        box_color,
        4
    )

    # Label
    text = (
        f"{label} | "
        f"Score: {score:.4f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.8

    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    # Background for text
    cv2.rectangle(
        image,
        (
            x1,
            y1
        ),
        (
            x1 + text_width + 20,
            y1 + text_height + baseline + 20
        ),
        box_color,
        -1
    )

    # Text
    cv2.putText(
        image,
        text,
        (
            x1 + 10,
            y1 + text_height + 10
        ),
        font,
        font_scale,
        (
            0,
            0,
            0
        ),
        thickness,
        cv2.LINE_AA
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_DIR.exists():

        raise RuntimeError(
            f"Input folder not found: "
            f"{INPUT_DIR}"
        )

    image_paths = []

    for image_path in INPUT_DIR.iterdir():

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in VALID_EXTENSIONS
        ):

            image_paths.append(
                image_path
            )

    image_paths.sort()

    print(
        f"\nImages Found: "
        f"{len(image_paths)}"
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            f"No images found in "
            f"{INPUT_DIR}"
        )

    monitor_count = 0
    anomaly_count = 0
    failed_count = 0

    results = []

    print(
        "\nAnnotating images..."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                raise RuntimeError(
                    "OpenCV could not read image"
                )

            result = predict(
                image
            )

            annotated = annotate_image(
                image.copy(),
                result
            )

            output_name = (
                f"{OUTPUT_PREFIX}_"
                f"{index:03d}.jpg"
            )

            output_path = (
                OUTPUT_DIR /
                output_name
            )

            success = cv2.imwrite(
                str(output_path),
                annotated
            )

            if not success:

                raise RuntimeError(
                    "Failed to save image"
                )

            if result["label"] == "MONITOR":

                monitor_count += 1

            else:

                anomaly_count += 1

            results.append({
                "original_file": image_path.name,
                "annotated_file": output_name,
                "label": result["label"],
                "anomaly_score": round(
                    result["anomaly_score"],
                    6
                ),
                "threshold": round(
                    THRESHOLD,
                    6
                ),
                "score_ratio": round(
                    result["score_ratio"],
                    4
                )
            })

        except Exception as e:

            failed_count += 1

            results.append({
                "original_file": image_path.name,
                "annotated_file": None,
                "label": "ERROR",
                "error": str(e)
            })

    # ========================================================
    # SAVE RESULTS JSON
    # ========================================================

    json_output = (
        OUTPUT_DIR /
        "annotation_results.json"
    )

    summary = {
        "total_images": len(image_paths),
        "monitor_count": monitor_count,
        "anomaly_count": anomaly_count,
        "failed_count": failed_count,
        "threshold": THRESHOLD,
        "output_directory": str(
            OUTPUT_DIR
        ),
        "predictions": results
    }

    with open(
        json_output,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n==================================="
    )

    print(
        "ANNOTATION COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Total Images : "
        f"{len(image_paths)}"
    )

    print(
        f"MONITOR      : "
        f"{monitor_count}"
    )

    print(
        f"ANOMALY      : "
        f"{anomaly_count}"
    )

    print(
        f"Failed       : "
        f"{failed_count}"
    )

    print(
        f"Output       : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"JSON         : "
        f"{json_output}"
    )

    print(
        "==================================="
    )


if __name__ == "__main__":

    main()