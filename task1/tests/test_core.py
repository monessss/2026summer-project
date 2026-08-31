"""Unit tests for dataset labels and detection matching."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from acceptance_test import required_class_counts, score_sample  # noqa: E402
from check_dataset import inspect_dataset, validate_label_line  # noqa: E402
from common import PROJECT_ROOT  # noqa: E402
from evaluate import box_iou, match_detections  # noqa: E402
from jetson_ros2_node import artifact_contract  # noqa: E402


def write_test_image(path: Path, pixel_value: int) -> None:
    image = np.full((8, 8, 3), pixel_value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to create test image: {path}")


class LabelValidationTests(unittest.TestCase):
    def test_valid_row(self) -> None:
        class_id, coordinates = validate_label_line(["2", "0.5", "0.5", "0.2", "0.4"], 3)
        self.assertEqual(class_id, 2)
        self.assertEqual(coordinates, (0.5, 0.5, 0.2, 0.4))

    def test_rejects_unknown_class(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_line(["3", "0.5", "0.5", "0.2", "0.4"], 3)

    def test_rejects_box_crossing_boundary(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_line(["0", "0.95", "0.5", "0.2", "0.4"], 3)


class MatchingTests(unittest.TestCase):
    def test_iou_identity(self) -> None:
        self.assertAlmostEqual(box_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)), 1.0)

    def test_correct_and_false_positive(self) -> None:
        ground_truth = [(0, (0.0, 0.0, 10.0, 10.0))]
        predictions = [
            (0, 0.9, (0.0, 0.0, 10.0, 10.0)),
            (2, 0.5, (20.0, 20.0, 30.0, 30.0)),
        ]
        result = match_detections(ground_truth, predictions, 0.5)
        self.assertEqual(len(result["correct"]), 1)
        self.assertEqual(result["false_positives"], [1])
        self.assertEqual(result["false_negatives"], [])

    def test_wrong_class_consumes_matching_ground_truth(self) -> None:
        result = match_detections(
            [(1, (0.0, 0.0, 10.0, 10.0))],
            [(2, 0.8, (0.0, 0.0, 10.0, 10.0))],
            0.5,
        )
        self.assertEqual(len(result["wrong_class"]), 1)
        self.assertEqual(result["false_negatives"], [])


class AcceptanceScoringTests(unittest.TestCase):
    def test_fixed_three_class_quota(self) -> None:
        self.assertEqual(required_class_counts([0, 1, 2]), {0: 7, 1: 7, 2: 6})

    def test_highest_confidence_detection_decides_result(self) -> None:
        predicted_class, confidence, correct = score_sample(
            [
                {"class_id": 2, "confidence": 0.70},
                {"class_id": 1, "confidence": 0.95},
            ],
            true_class_id=2,
        )
        self.assertEqual(predicted_class, 1)
        self.assertEqual(confidence, 0.95)
        self.assertFalse(correct)

    def test_no_detection_is_incorrect(self) -> None:
        self.assertEqual(score_sample([], true_class_id=0), (None, 0.0, False))


class JetsonArtifactContractTests(unittest.TestCase):
    def test_pytorch_and_tensorrt_contracts(self) -> None:
        self.assertEqual(
            artifact_contract(
                PROJECT_ROOT / "models" / "best.pt",
                PROJECT_ROOT / "results" / "videos" / "jetson_demo.mp4",
                PROJECT_ROOT / "results" / "jetson_detections.jsonl",
                PROJECT_ROOT / "results" / "fps_jetson.csv",
            ),
            "pytorch_fp16",
        )
        self.assertEqual(
            artifact_contract(
                PROJECT_ROOT / "models" / "best.engine",
                PROJECT_ROOT / "results" / "videos" / "jetson_demo_engine.mp4",
                PROJECT_ROOT / "results" / "jetson_detections_engine.jsonl",
                PROJECT_ROOT / "results" / "fps_jetson_engine.csv",
            ),
            "tensorrt_fp16",
        )

    def test_rejects_mixed_output_contract(self) -> None:
        self.assertEqual(
            artifact_contract(
                PROJECT_ROOT / "models" / "best.engine",
                PROJECT_ROOT / "results" / "videos" / "jetson_demo.mp4",
                PROJECT_ROOT / "results" / "jetson_detections.jsonl",
                PROJECT_ROOT / "results" / "fps_jetson.csv",
            ),
            "mismatch",
        )


class DatasetInspectionTests(unittest.TestCase):
    def test_minimal_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split_index, split in enumerate(("train", "val", "test")):
                images = root / "data" / "images" / split
                labels = root / "data" / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                write_test_image(images / f"sample_{split}.jpg", split_index * 80)
                (labels / f"sample_{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "path: data\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "nc: 1\n"
                "names: [keyboard]\n",
                encoding="utf-8",
            )
            report = inspect_dataset(data_yaml)
            self.assertTrue(report["valid"])
            self.assertEqual(report["splits"]["train"]["images"], 1)
            self.assertEqual(report["splits"]["test"]["instances"], 1)
            original_fingerprint = report["dataset_fingerprint_sha256"]
            train_label = root / "data" / "labels" / "train" / "sample_train.txt"
            train_label.write_bytes(b"0 0.5 0.5 0.2 0.2\r\n")
            self.assertEqual(
                inspect_dataset(data_yaml)["dataset_fingerprint_sha256"],
                original_fingerprint,
            )

    def test_rejects_capture_sequence_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split_index, split in enumerate(("train", "val", "test")):
                images = root / "data" / "images" / split
                labels = root / "data" / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                frame = 1 if split != "test" else 2
                filename = f"phone__DJI_20260826120000_0001_D_{frame:03d}_jpg.rf.hash{split}"
                write_test_image(images / f"{filename}.jpg", split_index * 80)
                (labels / f"{filename}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "path: data\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "nc: 1\n"
                "names: [phone]\n",
                encoding="utf-8",
            )
            report = inspect_dataset(data_yaml)
            self.assertFalse(report["valid"])
            self.assertEqual(report["capture_group_count"], 1)
            self.assertEqual(len(report["cross_split_capture_groups"]), 1)

if __name__ == "__main__":
    unittest.main()
