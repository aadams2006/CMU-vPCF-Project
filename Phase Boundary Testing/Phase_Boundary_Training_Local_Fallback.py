"""Deterministic local K-means benchmarks for the phase-boundary pipeline.

This file intentionally does not call its outputs DEC or IDEC. It validates
data/result plumbing, external metric logic, cluster-count diagnostics, seed
stability, and spatial boundary evaluation without requiring TensorFlow or the
uncommitted H5/DM3 research data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Results" / "Local_Fallback_Runs"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_inspection import create_cluster_inspection_report
from metrics import (
    aggregate_numeric_metrics,
    evaluate_clustering,
    optimal_label_mapping,
    pairwise_stability,
    remap_labels,
)


def generate_synthetic_phase_boundary_data() -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the original three-Gaussian clustering smoke test."""
    rng = np.random.default_rng(42)
    centers = np.array([[0.0, 0.0], [3.5, 3.0], [-2.5, 2.2]])
    features = []
    labels = []
    for cluster_idx, center in enumerate(centers):
        features.append(rng.normal(loc=center, scale=0.9, size=(140, 2)))
        labels.extend([cluster_idx] * 140)
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def generate_spatial_phase_benchmark(
    height: int = 48,
    width: int = 64,
    random_state: int = 314,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """Create three curved phases with noisy multivariate observations."""
    rng = np.random.default_rng(random_state)
    rows, cols = np.indices((height, width))
    y_coord = rows / max(height - 1, 1)
    x_coord = cols / max(width - 1, 1)
    left_boundary = 0.31 + 0.055 * np.sin(2.0 * np.pi * y_coord)
    right_boundary = 0.69 - 0.07 * np.cos(2.0 * np.pi * y_coord)

    label_grid = np.ones((height, width), dtype=np.int64)
    label_grid[x_coord < left_boundary] = 0
    label_grid[x_coord >= right_boundary] = 2
    labels = label_grid.reshape(-1)
    coordinates = np.column_stack([rows.reshape(-1), cols.reshape(-1)])

    phase_means = np.array(
        [
            [-2.0, 0.2, 1.2, 0.4, -1.1, 1.4],
            [0.9, 2.0, -0.7, -1.2, 0.9, 0.1],
            [2.8, -1.4, 1.9, 1.5, 0.1, -1.0],
        ]
    )
    features = phase_means[labels].copy()
    features[:, 0] += 0.45 * x_coord.reshape(-1)
    features[:, 1] += 0.35 * y_coord.reshape(-1)
    features += rng.normal(scale=0.62, size=features.shape)

    # Boundary observations are deliberately mixed so the benchmark exercises
    # boundary localization instead of producing a perfectly separable cartoon.
    distance_left = np.abs(x_coord - left_boundary)
    distance_right = np.abs(x_coord - right_boundary)
    near_left = distance_left.reshape(-1) < 0.022
    near_right = distance_right.reshape(-1) < 0.022
    features[near_left] = (
        0.58 * features[near_left]
        + 0.42 * phase_means[1 - labels[near_left]]
    )
    right_neighbour = np.where(labels[near_right] == 2, 1, 2)
    features[near_right] = (
        0.58 * features[near_right]
        + 0.42 * phase_means[right_neighbour]
    )
    return features, labels, (height, width), coordinates


def _csv_safe_record(record: Dict[str, object]) -> Dict[str, object]:
    output = {}
    for key, value in record.items():
        output[key] = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return output


def _fit_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> Tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    return model, model.fit_predict(features)


def _save_dataset(
    benchmark_dir: Path,
    features: np.ndarray,
    true_labels: np.ndarray,
    coordinates: Optional[np.ndarray],
) -> None:
    feature_frame = pd.DataFrame(
        features,
        columns=[f"feature_{idx}" for idx in range(features.shape[1])],
    )
    feature_frame.insert(0, "sample_idx", np.arange(features.shape[0]))
    feature_frame.to_csv(benchmark_dir / "features.csv", index=False)
    pd.DataFrame(
        {
            "sample_idx": np.arange(true_labels.size),
            "ground_truth": true_labels,
        }
    ).to_csv(benchmark_dir / "ground_truth_labels.csv", index=False)
    if coordinates is not None:
        pd.DataFrame(
            {
                "sample_idx": np.arange(coordinates.shape[0]),
                "row": coordinates[:, 0],
                "column": coordinates[:, 1],
            }
        ).to_csv(benchmark_dir / "spatial_coordinates.csv", index=False)


def _save_reference_plot(
    benchmark_dir: Path,
    benchmark_name: str,
    features: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    spatial_shape: Optional[Tuple[int, int]],
) -> None:
    mapping = optimal_label_mapping(true_labels, predicted_labels)
    matched = remap_labels(predicted_labels, mapping)
    if spatial_shape is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(
            features[:, 0], features[:, 1], c=true_labels, cmap="tab10", s=22, alpha=0.8
        )
        axes[0].set_title("Generating labels")
        axes[1].scatter(
            features[:, 0], features[:, 1], c=matched, cmap="tab10", s=22, alpha=0.8
        )
        axes[1].set_title("K-means labels (matched)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].imshow(true_labels.reshape(spatial_shape), cmap="tab10", interpolation="nearest")
        axes[0].set_title("Ground-truth phases")
        axes[1].imshow(matched.reshape(spatial_shape), cmap="tab10", interpolation="nearest")
        axes[1].set_title("K-means phases (matched)")
        for axis in axes:
            axis.set_axis_off()
    fig.suptitle(benchmark_name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(benchmark_dir / "reference_prediction_comparison.png", dpi=200)
    plt.close(fig)


def _run_cluster_count_sweep(
    features: np.ndarray,
    true_labels: np.ndarray,
    seeds: Sequence[int],
    cluster_counts: Sequence[int],
    spatial_shape: Optional[Tuple[int, int]],
) -> Tuple[list[Dict[str, object]], list[Dict[str, object]], Dict[str, object]]:
    rows = []
    aggregates = []
    for n_clusters in sorted(set(map(int, cluster_counts))):
        k_rows = []
        for seed in seeds:
            _, labels = _fit_kmeans(features, n_clusters, int(seed))
            metrics = evaluate_clustering(
                features,
                labels,
                y_true=true_labels,
                spatial_shape=spatial_shape,
                boundary_tolerance=1.0,
            )
            record = {"n_clusters_requested": n_clusters, "seed": int(seed), **metrics}
            rows.append(record)
            k_rows.append(record)
        for aggregate in aggregate_numeric_metrics(k_rows):
            aggregates.append({"n_clusters_requested": n_clusters, **aggregate})

    def best(metric: str, maximize: bool) -> Optional[int]:
        candidates = {
            int(row["n_clusters_requested"]): float(row["mean"])
            for row in aggregates
            if row["metric"] == metric
        }
        if not candidates:
            return None
        return (max if maximize else min)(candidates, key=candidates.get)

    recommendation = {
        "scope": "synthetic geometry diagnostic",
        "best_silhouette_k": best("silhouette_score", True),
        "best_davies_bouldin_k": best("davies_bouldin_score", False),
        "best_calinski_harabasz_k": best("calinski_harabasz_score", True),
        "warning": "These internal criteria do not establish semantic phase correctness.",
    }
    return rows, aggregates, recommendation


def _run_benchmark(
    run_dir: Path,
    benchmark_name: str,
    features: np.ndarray,
    true_labels: np.ndarray,
    seeds: Sequence[int],
    cluster_counts: Sequence[int],
    spatial_shape: Optional[Tuple[int, int]] = None,
    coordinates: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    benchmark_dir = run_dir / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    _save_dataset(benchmark_dir, features, true_labels, coordinates)

    truth_metrics = evaluate_clustering(
        features,
        true_labels,
        y_true=true_labels,
        spatial_shape=spatial_shape,
        boundary_tolerance=1.0,
    )
    pd.DataFrame([_csv_safe_record(truth_metrics)]).to_csv(
        benchmark_dir / "ground_truth_partition_metrics.csv",
        index=False,
    )

    metrics_rows = []
    label_runs: Dict[int, np.ndarray] = {}
    reference_labels = None
    for seed in seeds:
        seed_dir = benchmark_dir / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model, labels = _fit_kmeans(features, n_clusters=3, seed=int(seed))
        metrics = evaluate_clustering(
            features,
            labels,
            y_true=true_labels,
            spatial_shape=spatial_shape,
            boundary_tolerance=1.0,
        )
        record = {
            "algorithm": "sklearn.cluster.KMeans",
            "seed": int(seed),
            **metrics,
        }
        metrics_rows.append(record)
        label_runs[int(seed)] = labels
        if reference_labels is None:
            reference_labels = labels

        pd.DataFrame(
            {
                "sample_idx": np.arange(labels.size),
                "cluster_raw": labels,
                "cluster_matched": remap_labels(
                    labels,
                    optimal_label_mapping(true_labels, labels),
                ),
                "ground_truth": true_labels,
            }
        ).to_csv(seed_dir / "kmeans_cluster_labels.csv", index=False)
        centers = pd.DataFrame(
            model.cluster_centers_,
            columns=[f"feature_{idx}" for idx in range(features.shape[1])],
        )
        centers.insert(0, "cluster_raw", np.arange(model.n_clusters))
        centers.to_csv(seed_dir / "kmeans_fitted_cluster_centers.csv", index=False)
        pd.DataFrame([_csv_safe_record(record)]).to_csv(
            seed_dir / "kmeans_metrics.csv",
            index=False,
        )
        create_cluster_inspection_report(
            labels,
            model_name="kmeans",
            sample_names=np.array([f"sample_{idx:06d}" for idx in range(labels.size)]),
            ground_truth_labels=true_labels,
            save_dir=str(seed_dir),
        )

    pd.DataFrame([_csv_safe_record(row) for row in metrics_rows]).to_csv(
        benchmark_dir / "metrics_by_seed.csv",
        index=False,
    )
    aggregate_rows = aggregate_numeric_metrics(metrics_rows)
    pd.DataFrame(aggregate_rows).to_csv(benchmark_dir / "metrics_aggregate.csv", index=False)
    stability_rows = pairwise_stability(label_runs)
    pd.DataFrame(stability_rows).to_csv(benchmark_dir / "stability_pairwise.csv", index=False)
    pd.DataFrame(
        aggregate_numeric_metrics(
            stability_rows,
            excluded_keys=("seed_a", "seed_b"),
        )
    ).to_csv(benchmark_dir / "stability_aggregate.csv", index=False)

    sweep_rows, sweep_aggregate, recommendation = _run_cluster_count_sweep(
        features,
        true_labels,
        seeds,
        cluster_counts,
        spatial_shape,
    )
    pd.DataFrame([_csv_safe_record(row) for row in sweep_rows]).to_csv(
        benchmark_dir / "cluster_count_sweep_by_seed.csv",
        index=False,
    )
    pd.DataFrame(sweep_aggregate).to_csv(
        benchmark_dir / "cluster_count_sweep_aggregate.csv",
        index=False,
    )
    with open(benchmark_dir / "cluster_count_recommendation.json", "w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2)

    _save_reference_plot(
        benchmark_dir,
        benchmark_name,
        features,
        true_labels,
        np.asarray(reference_labels),
        spatial_shape,
    )
    return {
        "name": benchmark_name,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_ground_truth_clusters": int(np.unique(true_labels).size),
        "spatial_shape": spatial_shape,
        "metrics_by_seed": metrics_rows,
        "metrics_aggregate": aggregate_rows,
        "cluster_count_recommendation": recommendation,
    }


def _metric_mean(benchmark: Dict[str, object], metric: str) -> Optional[float]:
    for row in benchmark["metrics_aggregate"]:
        if row["metric"] == metric:
            return float(row["mean"])
    return None


def _write_analysis(run_dir: Path, benchmarks: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Local K-means benchmark analysis",
        "",
        "This run is a deterministic pipeline/evaluation benchmark. It does not train DEC or IDEC and does not use the research H5/DM3 data.",
        "",
    ]
    for benchmark in benchmarks:
        lines.extend(
            [
                f"## {str(benchmark['name']).replace('_', ' ').title()}",
                "",
                f"- Samples/features: {benchmark['n_samples']} / {benchmark['n_features']}",
                f"- Mean matched accuracy: {_metric_mean(benchmark, 'matched_accuracy'):.6f}",
                f"- Mean ARI: {_metric_mean(benchmark, 'adjusted_rand_index'):.6f}",
                f"- Mean NMI: {_metric_mean(benchmark, 'normalized_mutual_info'):.6f}",
                f"- Mean silhouette: {_metric_mean(benchmark, 'silhouette_score'):.6f}",
            ]
        )
        boundary_f1 = _metric_mean(benchmark, "boundary_f1")
        if boundary_f1 is not None:
            lines.extend(
                [
                    f"- Mean boundary F1 (1 px tolerance): {boundary_f1:.6f}",
                    f"- Mean exact boundary IoU: {_metric_mean(benchmark, 'boundary_iou_exact'):.6f}",
                    f"- Mean symmetric boundary distance: {_metric_mean(benchmark, 'mean_symmetric_boundary_distance_px'):.6f} px",
                ]
            )
        lines.extend(
            [
                f"- Internal-metric k choices: {benchmark['cluster_count_recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These results validate artifact saving, label matching, repeated-seed aggregation, cluster-count sweeps, and (for the spatial benchmark) boundary metrics. They are not evidence that DEC, IDEC, or K-means recovers phases in the real vPCF dataset.",
            "",
        ]
    )
    (run_dir / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def run_fallback_pipeline(
    output_root: Path | str | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cluster_counts: Sequence[int] = (2, 3, 4, 5, 6, 7),
) -> Path:
    """Run both local benchmarks and save every artifact inside the repository."""
    output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_kmeans_benchmark_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    blob_features, blob_labels = generate_synthetic_phase_boundary_data()
    spatial_features, spatial_labels, spatial_shape, coordinates = (
        generate_spatial_phase_benchmark()
    )
    benchmarks = [
        _run_benchmark(
            run_dir,
            "gaussian_blob_smoke_test",
            blob_features,
            blob_labels,
            seeds,
            cluster_counts,
        ),
        _run_benchmark(
            run_dir,
            "spatial_phase_boundary_test",
            spatial_features,
            spatial_labels,
            seeds,
            cluster_counts,
            spatial_shape=spatial_shape,
            coordinates=coordinates,
        ),
    ]
    run_summary = {
        "run_type": "local_kmeans_benchmark",
        "algorithm": "sklearn.cluster.KMeans",
        "deep_models_trained": False,
        "research_data_used": False,
        "seeds": list(map(int, seeds)),
        "cluster_count_candidates": list(map(int, cluster_counts)),
        "benchmarks": benchmarks,
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
    _write_analysis(run_dir, benchmarks)

    manifest_path = output_root / "phase_boundary_run_manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    manifest.append(
        {
            "run_dir": run_dir.name,
            "timestamp_utc": run_stamp,
            "run_type": "local_kmeans_benchmark",
            "algorithms": ["kmeans"],
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    (output_root / "latest_run.txt").write_text(run_dir.name, encoding="utf-8")
    print(f"Local K-means benchmark completed: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clearly labeled local K-means phase-boundary benchmarks."
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--cluster-counts",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7],
    )
    args = parser.parse_args()
    run_fallback_pipeline(
        output_root=args.output_root,
        seeds=args.seeds,
        cluster_counts=args.cluster_counts,
    )


if __name__ == "__main__":
    main()
