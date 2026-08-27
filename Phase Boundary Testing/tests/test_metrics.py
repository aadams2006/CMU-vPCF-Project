from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import (
    acc,
    aggregate_numeric_metrics,
    compute_boundary_metrics,
    compute_external_metrics,
    confusion_matrix_table,
    pairwise_stability,
)


class ClusteringMetricTests(unittest.TestCase):
    def test_permuted_cluster_ids_are_perfect(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])
        metrics, mapping = compute_external_metrics(truth, pred)
        self.assertEqual(mapping, {0: 1, 1: 2, 2: 0})
        self.assertAlmostEqual(acc(truth, pred), 1.0)
        self.assertAlmostEqual(metrics["matched_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["adjusted_rand_index"], 1.0)

    def test_confusion_matrix_uses_matched_ids(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])
        classes, matrix = confusion_matrix_table(truth, pred)
        np.testing.assert_array_equal(classes, [0, 1, 2])
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=int) * 2)

    def test_boundary_metrics_are_label_permutation_invariant(self):
        truth_grid = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 2, 2, 1],
                [0, 2, 2, 1],
            ]
        )
        pred_grid = np.choose(truth_grid, [2, 0, 1])
        metrics = compute_boundary_metrics(
            truth_grid.reshape(-1),
            pred_grid.reshape(-1),
            spatial_shape=truth_grid.shape,
            tolerance=0,
        )
        self.assertEqual(metrics["boundary_f1"], 1.0)
        self.assertEqual(metrics["boundary_iou_exact"], 1.0)
        self.assertEqual(metrics["mean_symmetric_boundary_distance_px"], 0.0)

    def test_shifted_boundary_is_penalized(self):
        truth = np.zeros((8, 8), dtype=int)
        truth[:, 4:] = 1
        pred = np.zeros((8, 8), dtype=int)
        pred[:, 5:] = 1
        metrics = compute_boundary_metrics(
            truth.reshape(-1),
            pred.reshape(-1),
            spatial_shape=truth.shape,
            tolerance=0,
        )
        self.assertLess(metrics["boundary_f1"], 1.0)
        self.assertGreater(metrics["mean_symmetric_boundary_distance_px"], 0.0)

    def test_pairwise_stability_is_label_invariant(self):
        rows = pairwise_stability(
            {
                1: np.array([0, 0, 1, 1]),
                2: np.array([1, 1, 0, 0]),
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["adjusted_rand_index"], 1.0)
        self.assertEqual(rows[0]["normalized_mutual_info"], 1.0)

    def test_aggregate_metrics_reports_confidence_interval(self):
        rows = aggregate_numeric_metrics(
            [{"seed": 1, "score": 0.8}, {"seed": 2, "score": 1.0}]
        )
        score = next(row for row in rows if row["metric"] == "score")
        self.assertAlmostEqual(score["mean"], 0.9)
        self.assertEqual(score["n"], 2)
        self.assertLess(score["ci_low"], score["mean"])
        self.assertGreater(score["ci_high"], score["mean"])


if __name__ == "__main__":
    unittest.main()
