#!/usr/bin/env python3
"""Train Task 1 detector and publish the selected best.pt artifact."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_dataset import inspect_dataset, print_report
from common import (
    PROJECT_ROOT,
    load_yaml,
    portable_path,
    require_ascii_project_path_on_windows,
    resolve_dataset,
    resolve_project_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", help="Training YAML relative to task1")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration without training")
    return parser.parse_args()


def resolve_model(value: str) -> str:
    candidate = resolve_project_path(value)
    return str(candidate) if candidate.is_file() else value


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)

    required = {"model", "data", "project", "name"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required training settings: {', '.join(missing)}")

    data_yaml = resolve_project_path(str(config["data"]))
    data, _split_paths = resolve_dataset(data_yaml)
    dataset_report = inspect_dataset(data_yaml)
    print_report(dataset_report)
    write_json(PROJECT_ROOT / "results" / "dataset_report.json", dataset_report)
    if not dataset_report["valid"]:
        raise RuntimeError("Dataset validation failed; fix the reported errors before training")

    project_dir = resolve_project_path(str(config["project"]))
    run_name = str(config["name"])
    planned_run_dir = project_dir / run_name

    resolved_config: dict[str, Any] = dict(config)
    resolved_config.update(
        {
            "model": resolve_model(str(config["model"])),
            "data": str(data_yaml),
            "project": str(project_dir),
            "name": run_name,
        }
    )
    print("Training configuration")
    for key, value in resolved_config.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        write_json(PROJECT_ROOT / "results" / "requested_config.json", resolved_config)
        print("Dry run completed; training was not started.")
        return 0

    require_ascii_project_path_on_windows()
    if planned_run_dir.exists():
        raise FileExistsError(
            f"Training run directory already exists: {planned_run_dir}. "
            "Archive it under a different name before starting the formal run."
        )

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -r requirements-train.txt") from exc

    model = YOLO(resolved_config.pop("model"))
    data_value = resolved_config.pop("data")
    results = model.train(data=data_value, **resolved_config)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", getattr(results, "save_dir", planned_run_dir))).resolve()
    write_json(save_dir / "requested_config.json", config)

    best_source = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    if not best_source.is_file():
        raise FileNotFoundError(f"Training finished without the best checkpoint: {best_source}")
    selected_source = best_source

    published_model = PROJECT_ROOT / "models" / "best.pt"
    published_model.parent.mkdir(parents=True, exist_ok=True)
    if selected_source.resolve() != published_model.resolve():
        shutil.copy2(selected_source, published_model)

    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures" / "training"
    figures_dir.mkdir(parents=True, exist_ok=True)
    published_artifacts: list[str] = []
    for filename in (
        "results.png",
        "labels.jpg",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
    ):
        source = save_dir / filename
        if source.is_file():
            target = figures_dir / filename
            shutil.copy2(source, target)
            published_artifacts.append(str(target.relative_to(PROJECT_ROOT)))
    training_csv = save_dir / "results.csv"
    if training_csv.is_file():
        target_csv = results_dir / "training_results.csv"
        shutil.copy2(training_csv, target_csv)
        published_artifacts.append(str(target_csv.relative_to(PROJECT_ROOT)))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": portable_path(selected_source),
        "published_model": portable_path(published_model),
        "sha256": sha256_file(published_model),
        "run_directory": portable_path(save_dir),
        "classes": data["names"],
        "dataset_fingerprint_sha256": dataset_report["dataset_fingerprint_sha256"],
        "training_config": config,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "ultralytics": ultralytics.__version__,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
            ),
        },
        "published_artifacts": published_artifacts,
    }
    write_json(PROJECT_ROOT / "models" / "model_info.json", metadata)
    print(f"Best model: {published_model}")
    print(f"SHA-256: {metadata['sha256']}")
    print("Next: python src/evaluate.py --model models/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
