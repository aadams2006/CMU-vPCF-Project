from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PHASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PHASE_DIR / "src"
for path in (PHASE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_phase_boundary_models as training
from Phase_Boundary_Training_Local_Fallback import run_fallback_pipeline
from vpcf_data_loader import VPCFDataset


class PipelineIntegrationTests(unittest.TestCase):
    def test_kmeans_pipeline_saves_complete_repeated_seed_artifacts(self):
        rng = np.random.default_rng(7)
        features = np.vstack(
            [rng.normal(-2, 0.2, size=(12, 2)), rng.normal(2, 0.2, size=(12, 2))]
        ).astype(np.float32)
        dataset = VPCFDataset(features=features)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            labels_path = Path(tmp) / "labels.csv"
            pd.DataFrame({"phase": [0] * 12 + [1] * 12}).to_csv(labels_path, index=False)
            with patch.object(training, "load_vpcf_file", return_value=dataset):
                result = training.run_pipeline(
                    h5_file="placeholder.h5",
                    ground_truth_file=str(labels_path),
                    ground_truth_column="phase",
                    model="kmeans",
                    n_clusters=None,
                    seeds=[1, 2],
                    cluster_counts=[2, 3],
                    output_dir=str(output),
                    verbose=False,
                )
            self.assertTrue((output / "metrics_by_seed.csv").is_file())
            self.assertTrue((output / "metrics_aggregate.csv").is_file())
            self.assertTrue((output / "stability_pairwise.csv").is_file())
            self.assertTrue((output / "cluster_count_sweep_by_seed.csv").is_file())
            self.assertTrue((output / "seed_1" / "kmeans" / "kmeans_fitted_cluster_centers.csv").is_file() is False)
            self.assertTrue((output / "seed_1" / "kmeans" / "kmeans_cluster_centers.csv").is_file())
            self.assertIn("summary_path", result)

    def test_local_fallback_never_labels_kmeans_as_deep_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_fallback_pipeline(
                output_root=tmp,
                seeds=[42, 43],
                cluster_counts=[2, 3, 4],
            )
            summary = (run_dir / "run_summary.json").read_text(encoding="utf-8")
            filenames = [path.name.lower() for path in run_dir.rglob("*")]
        self.assertIn('"deep_models_trained": false', summary.lower())
        self.assertFalse(any(name.startswith("dec_") or name.startswith("idec_") for name in filenames))


if __name__ == "__main__":
    unittest.main()
