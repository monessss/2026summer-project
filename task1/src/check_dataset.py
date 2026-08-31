#!/usr/bin/env python3
"""Validate a YOLO detection dataset before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    decode_image,
    image_files,
    labels_directory,
    portable_path,
    resolve_dataset,
    resolve_project_path,
    sha256_file,
    write_json,
)


CAPTURE_GROUP_PATTERN = re.compile(r"^(?P<group>.+)_D_\d+_jpg\.rf\.[^.]+$")


def capture_group_name(image_path: Path) -> str | None:
    """Return the DJI capture-sequence ID encoded in a Roboflow filename."""
    match = CAPTURE_GROUP_PATTERN.match(image_path.stem)
    return match.group("group") if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.yaml", help="Dataset YAML relative to task1")
    parser.add_argument(
        "--report", default="results/dataset_report.json", help="JSON report relative to task1"
    )
    return parser.parse_args()


def validate_label_line(
    parts: list[str],
    class_count: int,
) -> tuple[int, tuple[float, float, float, float]]:
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, found {len(parts)}")
    class_value = float(parts[0])
    class_id = int(class_value)
    if class_value != class_id or not 0 <= class_id < class_count:
        raise ValueError(f"invalid class ID {parts[0]}")
    coordinates = tuple(float(value) for value in parts[1:])
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError("coordinates must be normalized to [0, 1]")
    x_center, y_center, width, height = coordinates
    if width <= 0.0 or height <= 0.0:
        raise ValueError("box width and height must be positive")
    if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1.0 + 1e-6:
        raise ValueError("box crosses the horizontal image boundary")
    if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1.0 + 1e-6:
        raise ValueError("box crosses the vertical image boundary")
    return class_id, coordinates


def inspect_dataset(data_yaml: Path) -> dict[str, Any]:
    data, split_paths = resolve_dataset(data_yaml)
    class_names: dict[int, str] = data["names"]
    errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, Any] = {}
    image_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    capture_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    fingerprint_rows: list[str] = [
        "classes:"
        + json.dumps(class_names, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ]

    for split, images_dir in split_paths.items():
        labels_dir = labels_directory(images_dir)
        if not images_dir.is_dir():
            errors.append(f"{split}: image directory does not exist: {images_dir}")
            split_reports[split] = {"images": 0, "instances": 0, "class_counts": {}}
            continue
        if not labels_dir.is_dir():
            errors.append(f"{split}: label directory does not exist: {labels_dir}")

        images = image_files(images_dir)
        if not images:
            errors.append(f"{split}: no supported images found in {images_dir}")

        class_counts: Counter[int] = Counter()
        missing_labels = 0
        empty_labels = 0
        invalid_rows = 0
        unreadable_images = 0
        unreadable_labels = 0
        label_stems = {path.stem for path in labels_dir.glob("*.txt")} if labels_dir.is_dir() else set()
        image_stems = {path.stem for path in images}

        for image_path in images:
            image_digest = sha256_file(image_path)
            fingerprint_rows.append(f"{split}/images/{image_path.name}:{image_digest}")
            image_hashes[image_digest].append((split, image_path.name))
            if decode_image(image_path) is None:
                unreadable_images += 1
                errors.append(f"{split}: unreadable image: {image_path.name}")
            capture_group = capture_group_name(image_path)
            if capture_group:
                capture_groups[capture_group].append((split, image_path.name))
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels += 1
                errors.append(f"{split}: missing label for {image_path.name}")
            else:
                try:
                    normalized_label_text = label_path.read_text(encoding="utf-8-sig").replace(
                        "\r\n", "\n"
                    ).replace("\r", "\n")
                except (OSError, UnicodeError) as exc:
                    unreadable_labels += 1
                    errors.append(f"{split}: unreadable UTF-8 label {label_path.name}: {exc}")
                    fingerprint_rows.append(
                        f"{split}/labels/{label_path.name}:raw:{sha256_file(label_path)}"
                    )
                    continue
                label_digest = hashlib.sha256(normalized_label_text.encode("utf-8")).hexdigest()
                fingerprint_rows.append(
                    f"{split}/labels/{label_path.name}:{label_digest}"
                )
                nonempty_rows = 0
                for line_number, line in enumerate(normalized_label_text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    nonempty_rows += 1
                    try:
                        class_id, _coordinates = validate_label_line(line.split(), len(class_names))
                    except (TypeError, ValueError) as exc:
                        invalid_rows += 1
                        errors.append(f"{portable_path(label_path)}:{line_number}: {exc}")
                    else:
                        class_counts[class_id] += 1
                if nonempty_rows == 0:
                    empty_labels += 1

        orphan_labels = sorted(label_stems - image_stems)
        for stem in orphan_labels:
            errors.append(f"{split}: label has no matching image: {stem}.txt")

        missing_classes = [
            class_names[class_id] for class_id in class_names if class_counts[class_id] == 0
        ]
        if missing_classes:
            errors.append(f"{split}: classes without instances: {missing_classes}")

        split_reports[split] = {
            "images": len(images),
            "instances": sum(class_counts.values()),
            "class_counts": {
                class_names[class_id]: class_counts[class_id] for class_id in class_names
            },
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "orphan_labels": len(orphan_labels),
            "invalid_rows": invalid_rows,
            "unreadable_images": unreadable_images,
            "unreadable_labels": unreadable_labels,
            "images_dir": portable_path(images_dir),
            "labels_dir": portable_path(labels_dir),
        }

    duplicate_groups = []
    for digest, occurrences in image_hashes.items():
        occurrence_splits = {split for split, _name in occurrences}
        if len(occurrence_splits) > 1:
            duplicate_groups.append({"sha256": digest, "files": occurrences})
            errors.append(f"duplicate image appears across splits: {occurrences}")

    cross_split_capture_groups = []
    for group_name, occurrences in sorted(capture_groups.items()):
        occurrence_splits = {split for split, _name in occurrences}
        if len(occurrence_splits) > 1:
            cross_split_capture_groups.append({"capture_group": group_name, "files": occurrences})
            errors.append(
                f"continuous DJI capture group appears across splits: {group_name} "
                f"({sorted(occurrence_splits)})"
            )

    dataset_fingerprint = hashlib.sha256(
        ("\n".join(sorted(fingerprint_rows)) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "data_yaml": portable_path(data_yaml),
        "classes": class_names,
        "splits": split_reports,
        "duplicate_groups": duplicate_groups,
        "capture_group_count": len(capture_groups),
        "cross_split_capture_groups": cross_split_capture_groups,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def print_report(report: dict[str, Any]) -> None:
    print("Dataset inspection")
    for split, split_report in report["splits"].items():
        print(
            f"  {split:5s}: {split_report['images']:4d} images, "
            f"{split_report['instances']:4d} instances, "
            f"{split_report.get('invalid_rows', 0)} invalid rows"
        )
        for class_name, count in split_report.get("class_counts", {}).items():
            print(f"         {class_name}: {count}")
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    for error in report["errors"]:
        print(f"[ERROR] {error}")
    print("Dataset is valid." if report["valid"] else "Dataset validation failed.")


def main() -> int:
    args = parse_args()
    data_yaml = resolve_project_path(args.data)
    report_path = resolve_project_path(args.report)
    try:
        report = inspect_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as exc:
        report = {
            "data_yaml": str(data_yaml),
            "classes": {},
            "splits": {},
            "duplicate_groups": [],
            "capture_group_count": 0,
            "cross_split_capture_groups": [],
            "dataset_fingerprint_sha256": "",
            "errors": [str(exc)],
            "warnings": [],
            "valid": False,
        }
    write_json(report_path, report)
    print_report(report)
    print(f"Report: {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
