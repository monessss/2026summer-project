"""Unit tests for dataset labels and detection matching."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from check_dataset import inspect_dataset, validate_label_line  # noqa: E402
from evaluate import box_iou, match_detections  # noqa: E402


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


class DatasetInspectionTests(unittest.TestCase):
    def test_minimal_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for split_index, split in enumerate(("train", "val", "test")):
                images = root / "data" / "images" / split
                labels = root / "data" / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / f"sample_{split}.jpg").write_bytes(f"image-{split_index}".encode())
                (labels / f"sample_{split}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "path: data\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "nc: 2\n"
                "names: [keyboard, phone]\n",
                encoding="utf-8",
            )
            report = inspect_dataset(data_yaml)
            self.assertTrue(report["valid"])
            self.assertEqual(report["splits"]["train"]["images"], 1)
            self.assertEqual(report["splits"]["test"]["instances"], 1)

if __name__ == "__main__":
    unittest.main()
