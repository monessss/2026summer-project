#!/usr/bin/env python3
"""Validate a YOLO detection dataset before training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    image_files,
    labels_directory,
    resolve_dataset,
    resolve_project_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.yaml", help="Dataset YAML relative to task1")
    parser.add_argument(
        "--report", default="results/dataset_report.json", help="JSON report relative to task1"
    )
    parser.add_argument(
        "--skip-hash-duplicates",
        action="store_true",
        help="Skip slower cross-split duplicate-image hashing",
    )
    parser.add_argument(
        "--strict-missing-labels",
        action="store_true",
        help="Treat images without label files as errors instead of background-image warnings",
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


def inspect_dataset(
    data_yaml: Path,
    *,
    hash_duplicates: bool = True,
    strict_missing_labels: bool = False,
) -> dict[str, Any]:
    data, split_paths = resolve_dataset(data_yaml)
    class_names: dict[int, str] = data["names"]
    errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, Any] = {}
    image_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

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
        label_stems = {path.stem for path in labels_dir.glob("*.txt")} if labels_dir.is_dir() else set()
        image_stems = {path.stem for path in images}

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels += 1
                message = f"{split}: missing label for {image_path.name}"
                (errors if strict_missing_labels else warnings).append(message)
            else:
                nonempty_rows = 0
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    nonempty_rows += 1
                    try:
                        class_id, _coordinates = validate_label_line(line.split(), len(class_names))
                    except (TypeError, ValueError) as exc:
                        invalid_rows += 1
                        errors.append(f"{label_path}:{line_number}: {exc}")
                    else:
                        class_counts[class_id] += 1
                if nonempty_rows == 0:
                    empty_labels += 1

            if hash_duplicates:
                image_hashes[sha256_file(image_path)].append((split, image_path.name))

        orphan_labels = sorted(label_stems - image_stems)
        for stem in orphan_labels:
            warnings.append(f"{split}: label has no matching image: {stem}.txt")

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
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
        }

    duplicate_groups = []
    if hash_duplicates:
        for digest, occurrences in image_hashes.items():
            occurrence_splits = {split for split, _name in occurrences}
            if len(occurrence_splits) > 1:
                duplicate_groups.append({"sha256": digest, "files": occurrences})
                errors.append(f"duplicate image appears across splits: {occurrences}")

    return {
        "data_yaml": str(data_yaml.resolve()),
        "classes": class_names,
        "splits": split_reports,
        "duplicate_groups": duplicate_groups,
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
        report = inspect_dataset(
            data_yaml,
            hash_duplicates=not args.skip_hash_duplicates,
            strict_missing_labels=args.strict_missing_labels,
        )
    except (FileNotFoundError, ValueError) as exc:
        report = {
            "data_yaml": str(data_yaml),
            "classes": {},
            "splits": {},
            "duplicate_groups": [],
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
