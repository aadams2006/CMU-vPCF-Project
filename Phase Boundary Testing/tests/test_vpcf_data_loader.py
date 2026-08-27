from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vpcf_data_loader import (
    VPCFDataset,
    attach_ground_truth,
    load_ground_truth_labels,
)


class GroundTruthLoaderTests(unittest.TestCase):
    def test_loads_csv_label_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            pd.DataFrame({"sample": [0, 1, 2], "phase": [2, 0, 1]}).to_csv(path, index=False)
            labels, shape, metadata = load_ground_truth_labels(
                path,
                expected_samples=3,
                label_column="phase",
            )
        np.testing.assert_array_equal(labels, [2, 0, 1])
        self.assertIsNone(shape)
        self.assertEqual(metadata["label_column"], "phase")

    def test_loads_and_collapses_image_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.fromarray(np.array([[10, 10], [20, 30]], dtype=np.uint8)).save(path)
            labels, shape, _ = load_ground_truth_labels(
                path,
                expected_samples=4,
                label_map={"10": 0, "20": 1, "30": 1},
            )
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        self.assertEqual(shape, (2, 2))

    def test_reference_figure_size_mismatch_is_rejected(self):
        dataset = VPCFDataset(features=np.zeros((6, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference_with_legend.png"
            Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "aligned mask/vector"):
                attach_ground_truth(dataset, path)

    def test_dataset_validates_spatial_shape(self):
        with self.assertRaisesRegex(ValueError, "one cell per sample"):
            VPCFDataset(
                features=np.zeros((6, 2), dtype=np.float32),
                spatial_shape=(2, 2),
            )


if __name__ == "__main__":
    unittest.main()
