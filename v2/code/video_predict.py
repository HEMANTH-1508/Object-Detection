import json
from pathlib import Path

from ultralytics import YOLO

from config import (
    PROJECT_DIR,
    MODEL_DIR,
    ANNOTATED_DIR,
    JSON_DIR,
)


VIDEOS_DIR = PROJECT_DIR / "Ey-monitor-videos"

BEST_MODEL = (
    MODEL_DIR
    / "monitor_detector"
    / "weights"
    / "best.pt"
)

ANNOTATED_VIDEO_DIR = ANNOTATED_DIR / "videos"
JSON_VIDEO_DIR = JSON_DIR / "videos"


if not BEST_MODEL.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{BEST_MODEL}\n\n"
        "Run train.py first."
    )

if not VIDEOS_DIR.exists():
    raise FileNotFoundError(
        f"Video directory not found:\n{VIDEOS_DIR}"
    )

ANNOTATED_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

JSON_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


video_files = sorted(
    [
        file
        for file in VIDEOS_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in VIDEO_EXTENSIONS
    ]
)


if not video_files:
    raise FileNotFoundError(
        f"No video files found inside:\n{VIDEOS_DIR}\n\n"
        "Supported formats: "
        + ", ".join(sorted(VIDEO_EXTENSIONS))
    )

print("Loading trained model...")

model = YOLO(str(BEST_MODEL))

print(f"Model loaded successfully:")
print(f"  {BEST_MODEL}")

print()
print("=" * 60)
print(f"Found {len(video_files)} video(s)")
print("=" * 60)

for video_path in video_files:

    print()
    print("=" * 60)
    print(f"Processing video: {video_path.name}")
    print("=" * 60)

    video_name = video_path.stem

    output_video_path = (
        ANNOTATED_VIDEO_DIR
        / f"{video_name}_annotated.mp4"
    )

    output_json_path = (
        JSON_VIDEO_DIR
        / f"{video_name}.json"
    )

    print(f"Input video:")
    print(f"  {video_path}")

    print()
    print(f"Annotated video:")
    print(f"  {output_video_path}")

    print()
    print(f"JSON output:")
    print(f"  {output_json_path}")

    print()
    print("Starting frame-by-frame prediction...")

    frames_output = []

    results = model.predict(
        source=str(video_path),
        conf=0.25,
        imgsz=640,
        save=True,
        save_txt=False,
        show=False,
        project=str(ANNOTATED_VIDEO_DIR),
        name=video_name,
        exist_ok=True,
        stream=True,
    )


    for frame_number, result in enumerate(results, start=1):
        fps = 0

        if hasattr(result, "speed"):
            pass

        frame_detections = []


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

                x1, y1, x2, y2 = (
                    boxes.xyxy[i].tolist()
                )

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

                frame_detections.append(
                    detection
                )

        frame_output = {
            "frame": frame_number,

            "detections": frame_detections
        }


        frames_output.append(
            frame_output
        )

        print(
            f"Frame {frame_number}: "
            f"{len(frame_detections)} detection(s)"
        )


    output = {
        "video": video_path.name,

        "frames": frames_output
    }

    with open(
        output_json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


    generated_video_dir = (
        ANNOTATED_VIDEO_DIR
        / video_name
    )


    generated_video = None


    if generated_video_dir.exists():

        possible_videos = [
            file
            for file in generated_video_dir.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in VIDEO_EXTENSIONS
        ]


        if possible_videos:

            generated_video = possible_videos[0]

    if generated_video is not None:

        try:

            generated_video.replace(
                output_video_path
            )
            try:
                generated_video_dir.rmdir()
            except OSError:
                pass

        except Exception as error:

            print(
                "Warning: Could not move annotated video:"
            )

            print(error)

    else:

        print(
            "Warning: Annotated video was not found "
            "in the expected YOLO output directory."
        )

    print()
    print("-" * 60)
    print(f"Completed: {video_path.name}")
    print(f"Frames processed: {len(frames_output)}")
    print(f"JSON created: {output_json_path}")
    print(f"Annotated video: {output_video_path}")
    print("-" * 60)


print()
print("=" * 60)
print("VIDEO PREDICTION COMPLETED")
print("=" * 60)

print()
print(f"Input videos:")
print(f"  {VIDEOS_DIR}")

print()
print(f"Annotated videos:")
print(f"  {ANNOTATED_VIDEO_DIR}")

print()
print(f"JSON files:")
print(f"  {JSON_VIDEO_DIR}")

print()
print("=" * 60)