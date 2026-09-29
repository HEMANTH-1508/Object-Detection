import json
from pathlib import Path
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torchvision.transforms as T

TRAIN_DIR = Path("train")
MODEL_DIR = Path("../models/monitor_images")
OUTPUT_DIR = Path("../outputs/monitor_images")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
N_NEIGHBORS = 5

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("===================================")
print("MONITOR ANOMALY MODEL TRAINING")
print("===================================")
print(f"Device: {DEVICE}")

print("\nLoading EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0
)

model = model.to(DEVICE)
model.eval()

image_paths = []

for img_path in TRAIN_DIR.iterdir():
    if (
        img_path.is_file()
        and img_path.suffix.lower() in VALID_EXTENSIONS
    ):
        image_paths.append(img_path)

image_paths.sort()

print(f"Training Images Found: {len(image_paths)}")

if len(image_paths) < N_NEIGHBORS + 1:
    raise RuntimeError(
        f"Need at least {N_NEIGHBORS + 1} training images."
    )

embeddings = []
valid_image_paths = []
failed_images = []

print("\nExtracting Features...")

for img_path in tqdm(image_paths):
    try:
        image = Image.open(img_path).convert("RGB")

        tensor = transform(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = model(tensor)

        features = features.cpu().numpy().astype(np.float32)

        # L2 normalize
        features = features / (
            np.linalg.norm(
                features,
                axis=1,
                keepdims=True
            ) + 1e-10
        )

        embeddings.append(features[0])
        valid_image_paths.append(str(img_path))

    except Exception as e:
        failed_images.append({
            "file": str(img_path),
            "error": str(e)
        })

if len(embeddings) == 0:
    raise RuntimeError("No embeddings generated.")

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

print(f"\nEmbedding Shape: {embeddings.shape}")

print("\nTraining Nearest Neighbor Model...")

neighbor_count = min(
    N_NEIGHBORS + 1,
    len(embeddings)
)

nn_model = NearestNeighbors(
    n_neighbors=neighbor_count,
    metric="cosine"
)

nn_model.fit(embeddings)

# Query training images.
# First neighbor is the image itself.
distances, indices = nn_model.kneighbors(
    embeddings
)

# Remove self-distance.
neighbor_distances = distances[:, 1:]

train_scores = np.mean(
    neighbor_distances,
    axis=1
)

# Robust threshold calculation.
percentile_95 = float(
    np.percentile(train_scores, 95)
)

percentile_99 = float(
    np.percentile(train_scores, 99)
)

mean_score = float(
    np.mean(train_scores)
)

std_score = float(
    np.std(train_scores)
)

# Use 99th percentile as the initial conservative threshold.
threshold = percentile_99

print("\n===================================")
print("TRAINING SCORE STATISTICS")
print("===================================")
print(f"Minimum Score : {train_scores.min():.6f}")
print(f"Maximum Score : {train_scores.max():.6f}")
print(f"Mean Score    : {mean_score:.6f}")
print(f"Std Score     : {std_score:.6f}")
print(f"95% Threshold : {percentile_95:.6f}")
print(f"99% Threshold : {percentile_99:.6f}")
print(f"Used Threshold: {threshold:.6f}")
print("===================================")

model_path = MODEL_DIR / "nearest_neighbor_v2.pkl"
embedding_path = MODEL_DIR / "monitor_embeddings_v2.npy"
threshold_path = MODEL_DIR / "threshold_v2.json"

joblib.dump(
    nn_model,
    model_path
)

np.save(
    embedding_path,
    embeddings
)

threshold_data = {
    "threshold": threshold,
    "percentile_95": percentile_95,
    "percentile_99": percentile_99,
    "mean_score": mean_score,
    "std_score": std_score,
    "neighbors": N_NEIGHBORS,
    "image_size": IMAGE_SIZE,
    "feature_dimension": int(embeddings.shape[1]),
    "training_samples": int(len(embeddings)),
    "metric": "cosine",
    "model": "efficientnet_b0"
}

with open(
    threshold_path,
    "w"
) as f:
    json.dump(
        threshold_data,
        f,
        indent=4
    )

training_summary = {
    "device": DEVICE,
    "samples": int(len(embeddings)),
    "feature_dimension": int(embeddings.shape[1]),
    "failed_images": len(failed_images),
    "threshold": threshold,
    "percentile_95": percentile_95,
    "percentile_99": percentile_99,
    "mean_score": mean_score,
    "std_score": std_score,
    "image_size": IMAGE_SIZE,
    "neighbors": N_NEIGHBORS,
    "model": "efficientnet_b0",
    "metric": "cosine"
}

with open(
    OUTPUT_DIR / "training_summary_v2.json",
    "w"
) as f:
    json.dump(
        training_summary,
        f,
        indent=4
    )

with open(
    OUTPUT_DIR / "failed_images_v2.json",
    "w"
) as f:
    json.dump(
        failed_images,
        f,
        indent=4
    )

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")
print(f"Samples          : {len(embeddings)}")
print(f"Feature Dimension: {embeddings.shape[1]}")
print(f"Failed Images    : {len(failed_images)}")
print(f"Threshold        : {threshold:.6f}")
print(f"Model            : {model_path}")
print(f"Threshold File   : {threshold_path}")
print("===================================")