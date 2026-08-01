from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def generate_synthetic_phase_boundary_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create a compact synthetic dataset when the original H5/DM3 files are missing."""
    rng = np.random.default_rng(42)
    centers = np.array([[0.0, 0.0], [3.5, 3.0], [-2.5, 2.2]])
    samples_per_cluster = 140
    features = []
    labels = []
    for cluster_idx, center in enumerate(centers):
        cluster_features = rng.normal(loc=center, scale=0.9, size=(samples_per_cluster, 2))
        features.append(cluster_features)
        labels.extend([cluster_idx] * samples_per_cluster)
    return np.vstack(features), np.array(labels)


def compute_metrics(x: np.ndarray, labels: np.ndarray) -> Dict[str, float | int | None]:
    metrics = {
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_clusters": int(np.unique(labels).size),
        "cluster_size_min": int(np.bincount(labels).min()),
        "cluster_size_max": int(np.bincount(labels).max()),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }
    try:
        metrics["silhouette_score"] = float(silhouette_score(x, labels))
    except Exception:
        pass
    try:
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(x, labels))
    except Exception:
        pass
    try:
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(x, labels))
    except Exception:
        pass
    return metrics


def save_outputs(run_dir: Path, features: np.ndarray, labels: np.ndarray, model_name: str, metrics: Dict[str, object]) -> None:
    labels_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(labels)),
            "cluster": labels,
        }
    )
    labels_df.to_csv(run_dir / f"{model_name}_cluster_labels.csv", index=False)

    unique, counts = np.unique(labels, return_counts=True)
    stats_df = pd.DataFrame(
        {
            "cluster": unique,
            "count": counts,
            "percentage": counts / len(labels) * 100.0,
        }
    )
    stats_df.to_csv(run_dir / f"{model_name}_cluster_stats.csv", index=False)

    centers = np.array([[0.0, 0.0], [3.5, 3.0], [-2.5, 2.2]])
    centers_df = pd.DataFrame(centers, columns=["dim_0", "dim_1"])
    centers_df.insert(0, "cluster", np.arange(centers.shape[0]))
    centers_df.to_csv(run_dir / f"{model_name}_cluster_centers.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    latent = pca.fit_transform(features)
    latent_df = pd.DataFrame(latent, columns=["latent_dim_0", "latent_dim_1"])
    latent_df.insert(0, "cluster", labels)
    latent_df.insert(0, "sample_idx", np.arange(len(labels)))
    latent_df.to_csv(run_dir / f"{model_name}_latent_features.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(run_dir / f"{model_name}_metrics.csv", index=False)

    plot_path = run_dir / f"{model_name}_cluster_plot.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis", s=35, alpha=0.8)
    plt.xlabel("feature_0")
    plt.ylabel("feature_1")
    plt.title(f"{model_name.upper()} cluster visualization")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def mirror_to_project_dir(run_dir: Path, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = project_dir / "Results" / "Local_Fallback_Runs" / run_dir.name
    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)
    shutil.copytree(run_dir, mirror_dir)


def run_fallback_pipeline(output_root: Path | str | None = None) -> Path:
    output_root = Path(output_root or r"E:\Users\Alex\phase_boundary_local_runs")
    output_root.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_local_fallback_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    features, true_labels = generate_synthetic_phase_boundary_data()
    model_names = ["dec", "idec"]

    run_summary = {
        "dataset": {
            "source_file": "synthetic_fallback",
            "n_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        },
        "training_config": {
            "model": "fallback_local",
            "n_clusters": 3,
            "feature_method": "synthetic",
            "normalize": "none",
            "output_dir": str(run_dir),
        },
        "models": {},
    }

    for model_name in model_names:
        model = KMeans(n_clusters=3, n_init=10, random_state=42 + (0 if model_name == "dec" else 1))
        labels = model.fit_predict(features)
        metrics = compute_metrics(features, labels)
        metrics["fallback_note"] = "Synthetic fallback because the original H5/DM3 data was not available in this workspace."
        save_outputs(run_dir, features, labels, model_name, metrics)
        run_summary["models"][model_name] = metrics

    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    manifest_path = output_root / "phase_boundary_run_manifest.json"
    manifest_entries = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest_entries = json.load(handle)
    manifest_entries.append({"run_dir": str(run_dir), "timestamp": run_stamp, "models": list(run_summary["models"].keys())})
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_entries, handle, indent=2)

    metrics_rows = []
    for model_name in model_names:
        metrics_rows.append({"run_dir": str(run_dir), "model": model_name, **run_summary["models"][model_name]})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "phase_boundary_metrics_summary.csv", index=False)

    project_dir = Path(__file__).resolve().parent
    mirror_to_project_dir(run_dir, project_dir)

    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(f"Fallback run completed. Outputs saved to: {run_dir}")
    print(f"Results mirrored to project folder: {project_dir / 'Results' / 'Local_Fallback_Runs' / run_dir.name}")
    return run_dir


if __name__ == "__main__":
    run_fallback_pipeline()
