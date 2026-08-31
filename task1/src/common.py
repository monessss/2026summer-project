"""Shared helpers for Task 1 training, evaluation, and deployment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail with a useful message for invalid files."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the task1 project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def portable_path(path: str | Path) -> str:
    """Return a project-relative POSIX path when the target is inside task1."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def require_ascii_project_path_on_windows() -> None:
    """Fail early when a Windows path would break third-party image loaders."""
    if os.name == "nt":
        try:
            str(PROJECT_ROOT).encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "Formal Ultralytics training/evaluation on Windows requires an ASCII-only path. "
                "Clone the repository to C:\\yolo\\2026summer-project and run it there."
            ) from exc


def decode_image(path: Path) -> Any:
    """Decode an image from bytes so non-ASCII Windows paths work reliably."""
    import cv2
    import numpy as np

    try:
        encoded_image = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def encode_image(path: Path, image: Any) -> bool:
    """Encode and save an image through bytes for non-ASCII Windows paths."""
    import cv2

    success, encoded_image = cv2.imencode(path.suffix, image)
    if not success:
        return False
    try:
        encoded_image.tofile(str(path))
    except OSError:
        return False
    return True


def resolve_dataset(data_yaml: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Resolve YOLO dataset split paths and normalize class names."""
    data_yaml = data_yaml.resolve()
    data = load_yaml(data_yaml)

    root_value = data.get("path", ".")
    dataset_root = Path(str(root_value)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    split_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not split_value:
            raise ValueError(f"Missing '{split}' in {data_yaml}")
        split_path = Path(str(split_value)).expanduser()
        split_paths[split] = (
            split_path.resolve() if split_path.is_absolute() else (dataset_root / split_path).resolve()
        )

    names_value = data.get("names")
    if isinstance(names_value, list):
        names = {index: str(name) for index, name in enumerate(names_value)}
    elif isinstance(names_value, dict):
        names = {int(index): str(name) for index, name in names_value.items()}
    else:
        raise ValueError(f"'names' must be a list or mapping in {data_yaml}")
    data["names"] = dict(sorted(names.items()))

    declared_nc = int(data.get("nc", len(names)))
    if declared_nc != len(names):
        raise ValueError(f"nc={declared_nc}, but {len(names)} class names are defined")
    if set(names) != set(range(declared_nc)):
        raise ValueError("Class IDs must be consecutive integers beginning at 0")
    if len(set(names.values())) != len(names):
        raise ValueError("Class names must be unique")
    return data, split_paths


def image_files(directory: Path) -> list[Path]:
    """Return supported images in stable filename order."""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def labels_directory(images_directory: Path) -> Path:
    """Map a conventional .../images/<split> path to .../labels/<split>."""
    parts = list(images_directory.parts)
    try:
        image_index = max(index for index, part in enumerate(parts) if part == "images")
    except ValueError as exc:
        raise ValueError(f"Image split path must contain an 'images' directory: {images_directory}") from exc
    parts[image_index] = "labels"
    return Path(*parts)


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write human-readable UTF-8 JSON for experiment logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
