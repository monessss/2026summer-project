#!/usr/bin/env python3
"""Evaluate best.pt on the test split and save representative error cases."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Tuple

from common import (
    PROJECT_ROOT,
    image_files,
    labels_directory,
    resolve_dataset,
    resolve_project_path,
    write_json,
)


Box = Tuple[float, float, float, float]
GroundTruth = Tuple[int, Box]
Prediction = Tuple[int, float, Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-error-images", type=int, default=12)
    return parser.parse_args()


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(
    ground_truth: list[GroundTruth],
    predictions: list[Prediction],
    iou_threshold: float,
) -> dict[str, Any]:
    """Greedily match predictions by confidence and classify detection errors."""
    used_ground_truth: set[int] = set()
    correct: list[tuple[int, int, float]] = []
    wrong_class: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    prediction_order = sorted(range(len(predictions)), key=lambda index: -predictions[index][1])
    for prediction_index in prediction_order:
        predicted_class, _confidence, predicted_box = predictions[prediction_index]
        candidates = [
            (box_iou(predicted_box, gt_box), gt_index)
            for gt_index, (_gt_class, gt_box) in enumerate(ground_truth)
            if gt_index not in used_ground_truth
        ]
        if not candidates:
            false_positives.append(prediction_index)
            continue
        best_iou, best_gt_index = max(candidates)
        if best_iou < iou_threshold:
            false_positives.append(prediction_index)
            continue
        used_ground_truth.add(best_gt_index)
        if predicted_class == ground_truth[best_gt_index][0]:
            correct.append((prediction_index, best_gt_index, best_iou))
        else:
            wrong_class.append((prediction_index, best_gt_index, best_iou))

    false_negatives = [
        gt_index for gt_index in range(len(ground_truth)) if gt_index not in used_ground_truth
    ]
    return {
        "correct": correct,
        "wrong_class": wrong_class,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def read_ground_truth(label_path: Path, width: int, height: int) -> list[GroundTruth]:
    boxes: list[GroundTruth] = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id = int(float(parts[0]))
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        boxes.append(
            (
                class_id,
                (
                    (x_center - box_width / 2.0) * width,
                    (y_center - box_height / 2.0) * height,
                    (x_center + box_width / 2.0) * width,
                    (y_center + box_height / 2.0) * height,
                ),
            )
        )
    return boxes


def draw_box(image: Any, box: Box, color: tuple[int, int, int], text: str) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        text,
        (max(0, x1), max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def save_error_examples(
    model: Any,
    test_images_dir: Path,
    class_names: dict[int, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import cv2

    labels_dir = labels_directory(test_images_dir)
    error_dir = output_dir / "error_examples"
    error_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []

    test_images = image_files(test_images_dir)
    for image_number, image_path in enumerate(test_images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARNING] Cannot read {image_path}")
            continue
        height, width = image.shape[:2]
        ground_truth = read_ground_truth(labels_dir / f"{image_path.stem}.txt", width, height)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )[0]
        predictions: list[Prediction] = []
        if result.boxes is not None:
            for box in result.boxes:
                predictions.append(
                    (
                        int(box.cls[0].item()),
                        float(box.conf[0].item()),
                        tuple(float(value) for value in box.xyxy[0].detach().cpu().tolist()),
                    )
                )
        matching = match_detections(ground_truth, predictions, args.match_iou)
        fp_count = len(matching["false_positives"])
        fn_count = len(matching["false_negatives"])
        wrong_count = len(matching["wrong_class"])
        score = fp_count + 2 * fn_count + 3 * wrong_count
        if score:
            candidates.append(
                {
                    "score": score,
                    "image_path": image_path,
                    "image": image,
                    "ground_truth": ground_truth,
                    "predictions": predictions,
                    "matching": matching,
                    "false_positives": fp_count,
                    "false_negatives": fn_count,
                    "wrong_class": wrong_count,
                }
            )
        if image_number == 1 or image_number % 20 == 0 or image_number == len(test_images):
            print(f"Error analysis: {image_number}/{len(test_images)}")

    candidates.sort(key=lambda item: (-item["score"], item["image_path"].name))
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[: args.max_error_images], start=1):
        canvas = candidate["image"].copy()
        matching = candidate["matching"]
        missed_gt = set(matching["false_negatives"])
        wrong_gt = {gt_index for _pred_index, gt_index, _iou in matching["wrong_class"]}
        for gt_index, (class_id, box) in enumerate(candidate["ground_truth"]):
            color = (0, 165, 255) if gt_index in missed_gt or gt_index in wrong_gt else (0, 190, 0)
            draw_box(canvas, box, color, f"GT:{class_names.get(class_id, class_id)}")

        bad_predictions = set(matching["false_positives"])
        bad_predictions.update(index for index, _gt_index, _iou in matching["wrong_class"])
        for pred_index, (class_id, confidence, box) in enumerate(candidate["predictions"]):
            color = (0, 0, 220) if pred_index in bad_predictions else (220, 120, 0)
            draw_box(
                canvas,
                box,
                color,
                f"P:{class_names.get(class_id, class_id)} {confidence:.2f}",
            )

        header = (
            f"score={candidate['score']} FP={candidate['false_positives']} "
            f"FN={candidate['false_negatives']} CLS={candidate['wrong_class']}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (35, 35, 35), -1)
        cv2.putText(canvas, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output_path = error_dir / f"{rank:02d}_{candidate['image_path'].stem}.jpg"
        cv2.imwrite(str(output_path), canvas)
        rows.append(
            {
                "rank": rank,
                "image": candidate["image_path"].name,
                "error_score": candidate["score"],
                "false_positives": candidate["false_positives"],
                "false_negatives": candidate["false_negatives"],
                "wrong_class": candidate["wrong_class"],
                "ground_truth_count": len(candidate["ground_truth"]),
                "prediction_count": len(candidate["predictions"]),
                "output": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    csv_path = output_dir / "error_cases.csv"
    fieldnames = [
        "rank",
        "image",
        "error_score",
        "false_positives",
        "false_negatives",
        "wrong_class",
        "ground_truth_count",
        "prediction_count",
        "output",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "test_images": len(test_images),
        "images_with_errors": len(candidates),
        "saved_error_images": len(rows),
        "rows": rows,
    }
    write_json(output_dir / "error_summary.json", summary)
    return summary


def copy_metric_figures(source_dir: Path, output_dir: Path) -> list[str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("*.png", "*.jpg")
    copied: list[str] = []
    for pattern in patterns:
        for source in source_dir.glob(pattern):
            target = figures_dir / source.name
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(PROJECT_ROOT)))
    return sorted(set(copied))


def main() -> int:
    args = parse_args()
    model_path = resolve_project_path(args.model)
    data_yaml = resolve_project_path(args.data)
    output_dir = resolve_project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    data, split_paths = resolve_dataset(data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements-train.txt") from exc

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(output_dir),
        name="ultralytics_test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    metric_values: dict[str, Any] = {}
    for key, value in (getattr(metrics, "results_dict", {}) or {}).items():
        try:
            metric_values[key] = float(value)
        except (TypeError, ValueError):
            metric_values[key] = str(value)
    metric_values["speed_ms_per_image"] = getattr(metrics, "speed", {})
    metric_values["model"] = str(model_path)
    metric_values["data"] = str(data_yaml)
    metric_values["classes"] = data["names"]
    write_json(output_dir / "metrics.json", metric_values)

    evaluation_dir = Path(getattr(metrics, "save_dir", output_dir / "ultralytics_test"))
    copied_figures = copy_metric_figures(evaluation_dir, output_dir)
    error_summary = save_error_examples(
        model=model,
        test_images_dir=split_paths["test"],
        class_names=data["names"],
        output_dir=output_dir,
        args=args,
    )
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Figures copied: {len(copied_figures)}")
    print(f"Error examples saved: {error_summary['saved_error_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
