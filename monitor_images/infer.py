import json
import joblib
import torch
import timm
import numpy as np

from pathlib import Path
from PIL import Image
import torchvision.transforms as T


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("../models/monitor_images/nearest_neighbor_v2.pkl")
THRESHOLD_PATH = Path("../models/monitor_images/threshold_v2.json")

TEST_DIR = Path("../Lab_Image/main/test")
OUTPUT_DIR = Path("../outputs/monitor_images")
OUTPUT_PATH = OUTPUT_DIR / "test_predictions_v2.json"

IMAGE_SIZE = 224
N_NEIGHBORS = 5

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0
)

model = model.to(DEVICE)
model.eval()

print("Loading nearest-neighbor model...")

nn_model = joblib.load(
    MODEL_PATH
)

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


def extract_features(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():
        features = model(tensor)

    features = features.cpu().numpy().astype(
        np.float32
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


def predict(image_path):

    features = extract_features(
        image_path
    )

    distances, indices = nn_model.kneighbors(
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
        anomaly_score / THRESHOLD
    )

    return {
        "label": label,
        "anomaly_score": round(
            anomaly_score,
            6
        ),
        "threshold": round(
            THRESHOLD,
            6
        ),
        "score_ratio": round(
            score_ratio,
            4
        ),
        "nearest_distances": [
            round(float(distance), 6)
            for distance in distances[0]
        ]
    }


def main():

    if not TEST_DIR.exists():
        raise RuntimeError(
            f"Test folder not found: {TEST_DIR}"
        )

    image_paths = []

    for image_path in TEST_DIR.iterdir():

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
        f"\nTest Images Found: "
        f"{len(image_paths)}"
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No valid images found in {TEST_DIR}"
        )

    predictions = []

    failed_images = []

    print("\nPredicting...")

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            result = predict(
                image_path
            )

            prediction = {
                "file": image_path.name,
                "path": str(image_path),
                "label": result["label"],
                "anomaly_score": result[
                    "anomaly_score"
                ],
                "threshold": result[
                    "threshold"
                ],
                "score_ratio": result[
                    "score_ratio"
                ],
                "nearest_distances": result[
                    "nearest_distances"
                ]
            }

            predictions.append(
                prediction
            )

        except Exception as e:

            failed_images.append({
                "file": image_path.name,
                "path": str(image_path),
                "error": str(e)
            })

    monitor_count = sum(
        1
        for item in predictions
        if item["label"] == "MONITOR"
    )

    anomaly_count = sum(
        1
        for item in predictions
        if item["label"] == "ANOMALY"
    )

    output = {
        "model": "efficientnet_b0",
        "device": DEVICE,
        "threshold": THRESHOLD,
        "total_images": len(image_paths),
        "successful_predictions": len(
            predictions
        ),
        "failed_predictions": len(
            failed_images
        ),
        "monitor_count": monitor_count,
        "anomaly_count": anomaly_count,
        "predictions": predictions,
        "failed_images": failed_images
    }

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print("\n===================================")
    print("BATCH PREDICTION COMPLETE")
    print("===================================")
    print(
        f"Total Images       : "
        f"{len(image_paths)}"
    )
    print(
        f"Successful         : "
        f"{len(predictions)}"
    )
    print(
        f"Failed             : "
        f"{len(failed_images)}"
    )
    print(
        f"MONITOR            : "
        f"{monitor_count}"
    )
    print(
        f"ANOMALY            : "
        f"{anomaly_count}"
    )
    print(
        f"Threshold          : "
        f"{THRESHOLD:.6f}"
    )
    print(
        f"JSON Output        : "
        f"{OUTPUT_PATH}"
    )
    print("===================================")


if __name__ == "__main__":
    main()