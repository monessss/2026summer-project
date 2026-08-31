#!/usr/bin/env python3
"""Interactive 20-object acceptance test with auditable images and CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    encode_image,
    portable_path,
    resolve_dataset,
    resolve_project_path,
    sha256_file,
    write_json,
)


REQUIRED_SAMPLES = 20
REQUIRED_IMGSZ = 640
REQUIRED_CONFIDENCE = 0.50


def required_class_counts(class_ids: list[int]) -> dict[int, int]:
    """Distribute the 20 acceptance samples evenly across every model class."""
    if not class_ids:
        raise ValueError("The model does not define any classes")
    ordered_ids = sorted(class_ids)
    base, remainder = divmod(REQUIRED_SAMPLES, len(ordered_ids))
    return {
        class_id: base + (1 if position < remainder else 0)
        for position, class_id in enumerate(ordered_ids)
    }


def score_sample(
    detections: list[dict[str, Any]], true_class_id: int
) -> tuple[int | None, float, bool]:
    """Score one sample by its highest-confidence detection."""
    if not detections:
        return None, 0.0, False
    top_detection = max(detections, key=lambda item: float(item["confidence"]))
    predicted_class_id = int(top_detection["class_id"])
    confidence = float(top_detection["confidence"])
    return predicted_class_id, confidence, predicted_class_id == true_class_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="results/acceptance_test")
    return parser.parse_args()


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "timestamp",
        "true_class_id",
        "true_class",
        "predicted_class_id",
        "predicted_class",
        "predicted_confidence",
        "detections_json",
        "correct",
        "fps",
        "image",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    import cv2
    from ultralytics import YOLO

    model_path = resolve_project_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    output_dir = resolve_project_path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_20_objects.csv"

    model = YOLO(str(model_path))
    class_names = {int(index): str(name) for index, name in model.names.items()}
    expected_data, _split_paths = resolve_dataset(resolve_project_path(args.data))
    if class_names != expected_data["names"]:
        raise ValueError(
            f"Model classes {class_names} do not match dataset classes {expected_data['names']}"
        )
    for stale_image in images_dir.glob("*.jpg"):
        stale_image.unlink()
    csv_path.unlink(missing_ok=True)
    (output_dir / "summary.json").unlink(missing_ok=True)
    class_requirements = required_class_counts(list(class_names))
    recorded_class_counts = {class_id: 0 for class_id in class_names}
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    camera_properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    print("Acceptance test controls")
    for class_id, class_name in class_names.items():
        print(
            f"  Press {class_id} to record '{class_name}' "
            f"({class_requirements[class_id]} samples required)"
        )
    print("  Press q to stop and keep partial results")

    rows: list[dict[str, Any]] = []
    fps = 0.0
    previous_time = time.perf_counter()
    consecutive_read_failures = 0
    try:
        while len(rows) < REQUIRED_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    raise RuntimeError("Camera failed to provide 30 consecutive frames")
                continue
            consecutive_read_failures = 0
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            detections: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_names.get(class_id, str(class_id)),
                            "confidence": round(float(box.conf[0].item()), 4),
                        }
                    )

            current_time = time.perf_counter()
            instantaneous_fps = 1.0 / max(current_time - previous_time, 1e-9)
            previous_time = current_time
            fps = instantaneous_fps if fps == 0.0 else fps * 0.9 + instantaneous_fps * 0.1
            annotated = result.plot()
            cv2.putText(
                annotated,
                f"Recorded {len(rows)}/{REQUIRED_SAMPLES}  FPS {fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Task1 20-object acceptance test", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("0") <= key <= ord("9"):
                true_class_id = key - ord("0")
                if true_class_id not in class_names:
                    print(f"Unknown class ID: {true_class_id}")
                    continue
                if recorded_class_counts[true_class_id] >= class_requirements[true_class_id]:
                    print(f"Quota already complete for {class_names[true_class_id]}")
                    continue
                predicted_class_id, predicted_confidence, correct = score_sample(
                    detections, true_class_id
                )
                sample_id = len(rows) + 1
                status = "correct" if correct else "error"
                image_path = images_dir / f"{sample_id:02d}_{class_names[true_class_id]}_{status}.jpg"
                if not encode_image(image_path, annotated):
                    raise RuntimeError(f"Cannot save acceptance image: {image_path}")
                rows.append(
                    {
                        "id": sample_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "true_class_id": true_class_id,
                        "true_class": class_names[true_class_id],
                        "predicted_class_id": predicted_class_id,
                        "predicted_class": (
                            class_names[predicted_class_id]
                            if predicted_class_id is not None
                            else "no_detection"
                        ),
                        "predicted_confidence": predicted_confidence,
                        "detections_json": json.dumps(detections, ensure_ascii=False),
                        "correct": correct,
                        "fps": round(fps, 2),
                        "image": str(image_path.relative_to(PROJECT_ROOT)),
                    }
                )
                recorded_class_counts[true_class_id] += 1
                write_rows(csv_path, rows)
                print(
                    f"Recorded {sample_id}/{REQUIRED_SAMPLES}: "
                    f"{class_names[true_class_id]} -> {status}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    correct_count = sum(bool(row["correct"]) for row in rows)
    accuracy = correct_count / len(rows) if rows else 0.0
    quotas_complete = recorded_class_counts == class_requirements
    protocol_matches = (
        args.camera == 0
        and args.imgsz == REQUIRED_IMGSZ
        and args.conf == REQUIRED_CONFIDENCE
        and str(args.device) == "0"
        and camera_properties["width"] == args.width == 640
        and camera_properties["height"] == args.height == 480
    )
    summary = {
        "required_samples": REQUIRED_SAMPLES,
        "completed_samples": len(rows),
        "required_class_counts": {
            class_names[class_id]: count for class_id, count in class_requirements.items()
        },
        "completed_class_counts": {
            class_names[class_id]: count for class_id, count in recorded_class_counts.items()
        },
        "quotas_complete": quotas_complete,
        "protocol_matches": protocol_matches,
        "camera": camera_properties,
        "correct": correct_count,
        "incorrect": len(rows) - correct_count,
        "accuracy": accuracy,
        "acceptance_threshold": 0.80,
        "scoring_rule": "highest-confidence detection class equals the true class",
        "confidence_threshold": args.conf,
        "model": portable_path(model_path),
        "model_sha256": sha256_file(model_path),
        "classes": class_names,
        "passed": (
            len(rows) == REQUIRED_SAMPLES
            and quotas_complete
            and protocol_matches
            and accuracy >= 0.80
        ),
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"Completed {len(rows)}/{REQUIRED_SAMPLES}; correct={correct_count}; "
        f"accuracy={accuracy:.1%}; passed={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
