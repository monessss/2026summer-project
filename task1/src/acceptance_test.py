#!/usr/bin/env python3
"""Interactive 20-object acceptance test with auditable images and CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, resolve_project_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default="results/acceptance_test")
    return parser.parse_args()


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "timestamp",
        "true_class_id",
        "true_class",
        "predicted_classes",
        "matched_confidence",
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
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Acceptance test controls")
    for class_id, class_name in class_names.items():
        print(f"  Press {class_id} to record one '{class_name}' sample")
    print("  Press q to stop and keep partial results")

    rows: list[dict[str, Any]] = []
    fps = 0.0
    previous_time = time.perf_counter()
    try:
        while len(rows) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
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
                f"Recorded {len(rows)}/{args.samples}  FPS {fps:.1f}",
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
                matches = [item for item in detections if item["class_id"] == true_class_id]
                matched_confidence = max((item["confidence"] for item in matches), default=0.0)
                correct = bool(matches)
                sample_id = len(rows) + 1
                status = "correct" if correct else "error"
                image_path = images_dir / f"{sample_id:02d}_{class_names[true_class_id]}_{status}.jpg"
                cv2.imwrite(str(image_path), annotated)
                rows.append(
                    {
                        "id": sample_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "true_class_id": true_class_id,
                        "true_class": class_names[true_class_id],
                        "predicted_classes": json.dumps(detections, ensure_ascii=False),
                        "matched_confidence": matched_confidence,
                        "correct": correct,
                        "fps": round(fps, 2),
                        "image": str(image_path.relative_to(PROJECT_ROOT)),
                    }
                )
                write_rows(csv_path, rows)
                print(f"Recorded {sample_id}/{args.samples}: {class_names[true_class_id]} -> {status}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    correct_count = sum(bool(row["correct"]) for row in rows)
    accuracy = correct_count / len(rows) if rows else 0.0
    summary = {
        "requested_samples": args.samples,
        "completed_samples": len(rows),
        "correct": correct_count,
        "incorrect": len(rows) - correct_count,
        "accuracy": accuracy,
        "acceptance_threshold": 0.80,
        "passed": len(rows) == args.samples and accuracy >= 0.80,
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"Completed {len(rows)}/{args.samples}; correct={correct_count}; "
        f"accuracy={accuracy:.1%}; passed={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
