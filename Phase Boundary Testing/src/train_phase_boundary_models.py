"""Reproducible DEC, IDEC, and K-means phase-boundary evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_H5_FILE = PROJECT_DIR / "Data" / "SyntheticModel_HfO_80pm_vPCFs_65.h5"
DEFAULT_DM3_FILE = PROJECT_DIR / "Data" / "SyntheticModel_HfO_80pm_gaussian_HAADF.dm3"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Results" / "Full_Runs"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_inspection import create_cluster_inspection_report
from metrics import (
    aggregate_numeric_metrics,
    ari,
    boundary_map,
    evaluate_clustering,
    nmi,
    optimal_label_mapping,
    pairwise_stability,
    remap_labels,
)
from vpcf_data_loader import (
    VPCFDataset,
    attach_ground_truth,
    check_dependencies,
    combine_datasets,
    load_vpcf_file,
    print_file_info,
)


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _csv_safe_record(record: Mapping[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            output[key] = json.dumps(value, sort_keys=True, default=_to_serializable)
        else:
            output[key] = _to_serializable(value)
    return output


def set_global_seed(seed: int, include_tensorflow: bool = False) -> None:
    """Set Python, NumPy, and optionally TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_tensorflow:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for DEC/IDEC. Install repository requirements "
                "or run --model kmeans for the non-neural baseline."
            ) from exc
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass


def compute_clustering_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    boundary_tolerance: float = 1.0,
    ignore_label: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Compatibility wrapper around the complete evaluation suite."""
    metrics = evaluate_clustering(
        x,
        labels,
        y_true=y_true,
        spatial_shape=spatial_shape,
        boundary_tolerance=boundary_tolerance,
        ignore_label=ignore_label,
    )
    if verbose:
        print("\nClustering metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    return metrics


def get_model_dimensions(
    input_dim: int,
    n_clusters: int,
    hidden_dims: Optional[list] = None,
) -> list:
    """Build the encoder/decoder dimension list."""
    hidden_dims = hidden_dims or [500, 500, 2000]
    return [input_dim, *hidden_dims, n_clusters]


def train_dec_model(
    x: np.ndarray,
    y: Optional[np.ndarray],
    n_clusters: int,
    save_dir: str,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[object, np.ndarray]:
    """Train an actual Deep Embedded Clustering model."""
    set_global_seed(random_state, include_tensorflow=True)
    from dec import DEC

    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print(f"\nTraining DEC seed={random_state}, architecture={dims}")

    model = DEC(
        dims=dims,
        n_clusters=n_clusters,
        save_dir=save_dir,
        random_state=random_state,
    )
    model.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    model.compile(optimizer="sgd")
    labels = model.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size,
    )
    return model, labels


def train_idec_model(
    x: np.ndarray,
    y: Optional[np.ndarray],
    n_clusters: int,
    save_dir: str,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    gamma: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[object, np.ndarray]:
    """Train an actual Improved Deep Embedded Clustering model."""
    set_global_seed(random_state, include_tensorflow=True)
    from idec import IDEC

    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print(f"\nTraining IDEC seed={random_state}, architecture={dims}, gamma={gamma}")

    model = IDEC(
        dims=dims,
        n_clusters=n_clusters,
        gamma=gamma,
        save_dir=save_dir,
        random_state=random_state,
    )
    model.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    model.compile(optimizer="sgd")
    labels = model.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size,
    )
    return model, labels


def train_kmeans_model(
    x: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """Fit the plain K-means baseline in the original feature space."""
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    return model, model.fit_predict(x)


def _model_artifacts(model_name: str, model, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    if model_name == "kmeans":
        return model.cluster_centers_, np.asarray(x), "original_features"
    return model.get_cluster_centers(), model.extract_features(x), "learned_encoder_embedding"


def _save_spatial_comparison(
    labels: np.ndarray,
    dataset: VPCFDataset,
    save_dir: Path,
    model_name: str,
    ignore_label: Optional[int],
) -> None:
    if dataset.labels is None or dataset.spatial_shape is None:
        return
    mapping = optimal_label_mapping(dataset.labels, labels, ignore_label=ignore_label)
    matched = remap_labels(labels, mapping)
    truth_grid = dataset.labels.reshape(dataset.spatial_shape)
    pred_grid = matched.reshape(dataset.spatial_shape)
    valid = np.ones(dataset.n_samples, dtype=bool)
    if ignore_label is not None:
        valid &= dataset.labels != ignore_label
    valid_grid = valid.reshape(dataset.spatial_shape)
    true_boundary = boundary_map(truth_grid, valid_grid)
    pred_boundary = boundary_map(pred_grid, valid_grid)

    pd.DataFrame(truth_grid).to_csv(save_dir / "ground_truth_phase_map.csv", index=False)
    pd.DataFrame(pred_grid).to_csv(save_dir / f"{model_name}_matched_phase_map.csv", index=False)
    pd.DataFrame(true_boundary.astype(int)).to_csv(
        save_dir / "ground_truth_boundary_map.csv",
        index=False,
    )
    pd.DataFrame(pred_boundary.astype(int)).to_csv(
        save_dir / f"{model_name}_boundary_map.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(truth_grid, cmap="tab10", interpolation="nearest")
    axes[0].set_title("Ground truth")
    axes[1].imshow(pred_grid, cmap="tab10", interpolation="nearest")
    axes[1].set_title(f"{model_name.upper()} matched")
    axes[2].imshow(truth_grid, cmap="gray", interpolation="nearest", alpha=0.35)
    axes[2].contour(true_boundary.astype(float), levels=[0.5], colors="lime", linewidths=1)
    axes[2].contour(pred_boundary.astype(float), levels=[0.5], colors="magenta", linewidths=1)
    axes[2].set_title("Boundaries: truth=green, pred=magenta")
    for axis in axes:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_dir / f"{model_name}_spatial_comparison.png", dpi=200)
    plt.close(fig)


def save_results(
    labels: np.ndarray,
    dataset: VPCFDataset,
    save_dir: str,
    model_name: str,
    model,
    metrics: Dict[str, object],
    training_config: Dict[str, object],
    ignore_label: Optional[int] = None,
) -> None:
    """Save fitted centers, assignments, embeddings, metrics, and audits."""
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).reshape(-1)
    labels_data: Dict[str, object] = {
        "sample_idx": np.arange(len(labels)),
        "sample_name": dataset.sample_names,
        "cluster_raw": labels,
    }
    if dataset.labels is not None:
        mapping = optimal_label_mapping(dataset.labels, labels, ignore_label=ignore_label)
        labels_data["cluster_matched"] = remap_labels(labels, mapping)
        labels_data["ground_truth"] = dataset.labels
    pd.DataFrame(labels_data).to_csv(output / f"{model_name}_cluster_labels.csv", index=False)

    unique, counts = np.unique(labels, return_counts=True)
    pd.DataFrame(
        {
            "cluster_raw": unique,
            "count": counts,
            "percentage": counts / len(labels) * 100.0,
        }
    ).to_csv(output / f"{model_name}_cluster_stats.csv", index=False)

    centers, embedding, embedding_type = _model_artifacts(model_name, model, dataset.features)
    centers_df = pd.DataFrame(
        centers,
        columns=[f"dim_{idx}" for idx in range(centers.shape[1])],
    )
    centers_df.insert(0, "cluster_raw", np.arange(centers.shape[0]))
    centers_df.to_csv(output / f"{model_name}_cluster_centers.csv", index=False)

    embedding_df = pd.DataFrame(
        embedding,
        columns=[f"embedding_dim_{idx}" for idx in range(embedding.shape[1])],
    )
    embedding_df.insert(0, "cluster_raw", labels)
    embedding_df.insert(0, "sample_name", dataset.sample_names)
    embedding_df.insert(0, "sample_idx", np.arange(len(labels)))
    embedding_df.to_csv(output / f"{model_name}_embedding.csv", index=False)
    pd.DataFrame([_csv_safe_record(metrics)]).to_csv(
        output / f"{model_name}_metrics.csv",
        index=False,
    )

    metadata = {
        "model_name": model_name,
        "algorithm": {
            "kmeans": "sklearn.cluster.KMeans",
            "dec": "Deep Embedded Clustering",
            "idec": "Improved Deep Embedded Clustering",
        }[model_name],
        "embedding_type": embedding_type,
        "cluster_center_space": embedding_type,
        "source_file": dataset.source_file,
        "ground_truth_source": dataset.ground_truth_source,
        "n_samples": len(labels),
        "training_config": training_config,
        "metrics": metrics,
    }
    with open(output / f"{model_name}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_to_serializable)

    create_cluster_inspection_report(
        predicted_labels=labels,
        model_name=model_name,
        sample_names=dataset.sample_names,
        ground_truth_labels=dataset.labels,
        save_dir=str(output),
        ignore_label=ignore_label,
    )
    _save_spatial_comparison(labels, dataset, output, model_name, ignore_label)


def run_cluster_count_sweep(
    x: np.ndarray,
    cluster_counts: Sequence[int],
    seeds: Sequence[int],
    y_true: Optional[np.ndarray] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    boundary_tolerance: float = 1.0,
    ignore_label: Optional[int] = None,
) -> Tuple[list[Dict[str, object]], list[Dict[str, object]], Dict[str, object]]:
    """Evaluate K-means across candidate k values without claiming semantic truth."""
    per_seed: list[Dict[str, object]] = []
    aggregate: list[Dict[str, object]] = []
    for n_clusters in sorted(set(map(int, cluster_counts))):
        if n_clusters < 2 or n_clusters >= x.shape[0]:
            raise ValueError(f"Invalid cluster-count candidate: {n_clusters}")
        k_records = []
        for seed in seeds:
            _, labels = train_kmeans_model(x, n_clusters=n_clusters, random_state=int(seed))
            metrics = evaluate_clustering(
                x,
                labels,
                y_true=y_true,
                spatial_shape=spatial_shape,
                boundary_tolerance=boundary_tolerance,
                ignore_label=ignore_label,
            )
            record = {"n_clusters_requested": n_clusters, "seed": int(seed), **metrics}
            per_seed.append(record)
            k_records.append(record)
        for row in aggregate_numeric_metrics(k_records):
            aggregate.append({"n_clusters_requested": n_clusters, **row})

    def metric_means(metric: str) -> Dict[int, float]:
        return {
            int(row["n_clusters_requested"]): float(row["mean"])
            for row in aggregate
            if row["metric"] == metric
        }

    silhouette = metric_means("silhouette_score")
    davies = metric_means("davies_bouldin_score")
    calinski = metric_means("calinski_harabasz_score")
    recommendation: Dict[str, object] = {
        "scope": "internal-geometry heuristic only",
        "warning": (
            "Internal metrics measure compactness/separation and do not establish "
            "semantic phase correctness."
        ),
        "best_silhouette_k": max(silhouette, key=silhouette.get) if silhouette else None,
        "best_davies_bouldin_k": min(davies, key=davies.get) if davies else None,
        "best_calinski_harabasz_k": max(calinski, key=calinski.get) if calinski else None,
    }
    return per_seed, aggregate, recommendation


def _resolve_model_names(model: str) -> Tuple[str, ...]:
    if model == "both":
        return ("dec", "idec")
    if model == "all":
        return ("kmeans", "dec", "idec")
    return (model,)


def _save_dataset_artifacts(
    dataset: VPCFDataset,
    output_dir: Path,
    ignore_label: Optional[int],
) -> None:
    metadata = {
        "source_file": dataset.source_file,
        "ground_truth_source": dataset.ground_truth_source,
        "n_samples": dataset.n_samples,
        "feature_dim": dataset.feature_dim,
        "spatial_shape": dataset.spatial_shape,
        "has_ground_truth": dataset.labels is not None,
        "ignore_label": ignore_label,
        "metadata": dataset.metadata,
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_to_serializable)
    if dataset.labels is not None:
        pd.DataFrame(
            {
                "sample_idx": np.arange(dataset.n_samples),
                "sample_name": dataset.sample_names,
                "ground_truth": dataset.labels,
                "is_evaluated": (
                    np.ones(dataset.n_samples, dtype=bool)
                    if ignore_label is None
                    else dataset.labels != ignore_label
                ),
            }
        ).to_csv(output_dir / "ground_truth_labels.csv", index=False)


def run_pipeline(
    h5_file: Optional[str] = None,
    dm3_file: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    ground_truth_dataset: Optional[str] = None,
    ground_truth_column: Optional[str] = None,
    label_map: Optional[Dict[str, int]] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    model: str = "all",
    n_clusters: Optional[int] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cluster_counts: Optional[Sequence[int]] = (2, 3, 4, 5, 6, 7),
    allow_cluster_mismatch: bool = False,
    ignore_label: Optional[int] = None,
    boundary_tolerance: float = 1.0,
    feature_method: str = "flatten",
    normalize: Optional[str] = "minmax",
    downsample_factor: Optional[int] = None,
    max_frames: Optional[int] = None,
    hidden_dims: Optional[list] = None,
    pretrain_epochs: int = 50,
    maxiter: int = 8000,
    update_interval: int = 140,
    batch_size: int = 256,
    gamma: float = 0.1,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    verbose: bool = True,
) -> Dict[str, object]:
    """Run identical-data baselines and deep models across repeated seeds."""
    if h5_file is None and dm3_file is None:
        raise ValueError("At least one of h5_file or dm3_file must be provided.")
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("At least one random seed is required.")

    dependencies = check_dependencies()
    if verbose:
        print("Checking dependencies:")
        for name, available in dependencies.items():
            print(f"  {'[YES]' if available else '[NO]'} {name}")

    datasets = []
    if h5_file is not None:
        if verbose:
            print_file_info(h5_file)
        datasets.append(
            load_vpcf_file(
                h5_file,
                feature_method=feature_method,
                normalize=normalize,
                downsample_factor=downsample_factor,
                max_frames=max_frames,
                verbose=verbose,
            )
        )
    if dm3_file is not None:
        if verbose:
            print_file_info(dm3_file)
        datasets.append(
            load_vpcf_file(
                dm3_file,
                feature_method=feature_method,
                normalize=normalize,
                downsample_factor=downsample_factor,
                verbose=verbose,
            )
        )

    dataset = combine_datasets(*datasets) if len(datasets) > 1 else datasets[0]
    if ground_truth_file is not None:
        dataset = attach_ground_truth(
            dataset,
            ground_truth_file,
            dataset_key=ground_truth_dataset,
            label_column=ground_truth_column,
            label_map=label_map,
            spatial_shape=spatial_shape,
        )
    elif spatial_shape is not None:
        raise ValueError("spatial_shape is only meaningful with aligned ground truth.")

    truth_cluster_count = None
    if dataset.labels is not None:
        valid_truth = dataset.labels
        if ignore_label is not None:
            valid_truth = valid_truth[valid_truth != ignore_label]
        truth_cluster_count = int(np.unique(valid_truth).size)
        if truth_cluster_count < 2:
            raise ValueError("Ground truth must contain at least two evaluated classes.")
    if n_clusters is None:
        if truth_cluster_count is None:
            raise ValueError(
                "Unlabeled data requires an explicit n_clusters value. Use the saved "
                "cluster-count sweep as a geometry heuristic, not semantic ground truth."
            )
        n_clusters = truth_cluster_count
    n_clusters = int(n_clusters)
    if (
        truth_cluster_count is not None
        and n_clusters != truth_cluster_count
        and not allow_cluster_mismatch
    ):
        raise ValueError(
            f"Requested k={n_clusters}, but ground truth contains {truth_cluster_count} "
            "evaluated classes. Pass the intended label map or explicitly allow the mismatch."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_dataset_artifacts(dataset, output, ignore_label)
    x, y = dataset.features, dataset.labels
    model_names = _resolve_model_names(model)
    training_config = {
        "h5_file": h5_file,
        "dm3_file": dm3_file,
        "ground_truth_file": ground_truth_file,
        "ground_truth_dataset": ground_truth_dataset,
        "ground_truth_column": ground_truth_column,
        "label_map": label_map,
        "model": model,
        "resolved_models": model_names,
        "n_clusters": n_clusters,
        "seeds": seeds,
        "cluster_counts": cluster_counts,
        "allow_cluster_mismatch": allow_cluster_mismatch,
        "ignore_label": ignore_label,
        "boundary_tolerance": boundary_tolerance,
        "feature_method": feature_method,
        "normalize": normalize,
        "downsample_factor": downsample_factor,
        "max_frames": max_frames,
        "hidden_dims": hidden_dims or [500, 500, 2000],
        "pretrain_epochs": pretrain_epochs,
        "maxiter": maxiter,
        "update_interval": update_interval,
        "batch_size": batch_size,
        "gamma": gamma,
    }

    if cluster_counts:
        sweep_rows, sweep_aggregate, recommendation = run_cluster_count_sweep(
            x,
            cluster_counts=cluster_counts,
            seeds=seeds,
            y_true=y,
            spatial_shape=dataset.spatial_shape,
            boundary_tolerance=boundary_tolerance,
            ignore_label=ignore_label,
        )
        pd.DataFrame([_csv_safe_record(row) for row in sweep_rows]).to_csv(
            output / "cluster_count_sweep_by_seed.csv",
            index=False,
        )
        pd.DataFrame(sweep_aggregate).to_csv(
            output / "cluster_count_sweep_aggregate.csv",
            index=False,
        )
        with open(output / "cluster_count_recommendation.json", "w", encoding="utf-8") as handle:
            json.dump(recommendation, handle, indent=2)

    metrics_by_seed: list[Dict[str, object]] = []
    labels_by_model: Dict[str, Dict[int, np.ndarray]] = {
        model_name: {} for model_name in model_names
    }
    results: Dict[str, object] = {"dataset": dataset, "runs": {}}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        results["runs"][seed_key] = {}
        for model_name in model_names:
            model_dir = output / seed_key / model_name
            if model_name == "kmeans":
                fitted_model, labels = train_kmeans_model(x, n_clusters, seed)
            elif model_name == "dec":
                fitted_model, labels = train_dec_model(
                    x=x,
                    y=y,
                    n_clusters=n_clusters,
                    save_dir=str(model_dir),
                    hidden_dims=hidden_dims,
                    pretrain_epochs=pretrain_epochs,
                    maxiter=maxiter,
                    update_interval=update_interval,
                    batch_size=batch_size,
                    random_state=seed,
                    verbose=verbose,
                )
            else:
                fitted_model, labels = train_idec_model(
                    x=x,
                    y=y,
                    n_clusters=n_clusters,
                    save_dir=str(model_dir),
                    hidden_dims=hidden_dims,
                    pretrain_epochs=pretrain_epochs,
                    maxiter=maxiter,
                    update_interval=update_interval,
                    batch_size=batch_size,
                    gamma=gamma,
                    random_state=seed,
                    verbose=verbose,
                )

            metrics = compute_clustering_metrics(
                x,
                labels,
                y_true=y,
                spatial_shape=dataset.spatial_shape,
                boundary_tolerance=boundary_tolerance,
                ignore_label=ignore_label,
                verbose=verbose,
            )
            record = {"model": model_name, "seed": seed, **metrics}
            metrics_by_seed.append(record)
            labels_by_model[model_name][seed] = np.asarray(labels)
            per_run_config = {**training_config, "seed": seed, "algorithm": model_name}
            save_results(
                labels,
                dataset,
                str(model_dir),
                model_name,
                fitted_model,
                metrics,
                per_run_config,
                ignore_label=ignore_label,
            )
            results["runs"][seed_key][model_name] = {
                "labels": labels,
                "metrics": metrics,
                "output_dir": str(model_dir),
            }

    pd.DataFrame([_csv_safe_record(row) for row in metrics_by_seed]).to_csv(
        output / "metrics_by_seed.csv",
        index=False,
    )
    aggregate_rows = []
    stability_rows = []
    stability_aggregate_rows = []
    for model_name in model_names:
        model_records = [row for row in metrics_by_seed if row["model"] == model_name]
        for row in aggregate_numeric_metrics(model_records):
            aggregate_rows.append({"model": model_name, **row})
        model_stability = pairwise_stability(labels_by_model[model_name])
        for row in model_stability:
            stability_rows.append({"model": model_name, **row})
        for row in aggregate_numeric_metrics(
            model_stability,
            excluded_keys=("seed_a", "seed_b"),
        ):
            stability_aggregate_rows.append({"model": model_name, **row})

    pd.DataFrame(aggregate_rows).to_csv(output / "metrics_aggregate.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output / "stability_pairwise.csv", index=False)
    pd.DataFrame(stability_aggregate_rows).to_csv(
        output / "stability_aggregate.csv",
        index=False,
    )

    agreement_rows = []
    for seed in seeds:
        for model_a, model_b in combinations(model_names, 2):
            labels_a = labels_by_model[model_a][seed]
            labels_b = labels_by_model[model_b][seed]
            agreement_rows.append(
                {
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "adjusted_rand_index": float(ari(labels_a, labels_b)),
                    "normalized_mutual_info": float(nmi(labels_a, labels_b)),
                }
            )
    pd.DataFrame(agreement_rows).to_csv(output / "cross_model_agreement.csv", index=False)

    summary = {
        "scope": (
            "full DEC/IDEC/K-means evaluation" if set(model_names) != {"kmeans"}
            else "K-means baseline evaluation"
        ),
        "dataset": {
            "source_file": dataset.source_file,
            "ground_truth_source": dataset.ground_truth_source,
            "n_samples": dataset.n_samples,
            "feature_dim": dataset.feature_dim,
            "spatial_shape": dataset.spatial_shape,
            "ground_truth_clusters": truth_cluster_count,
        },
        "training_config": training_config,
        "artifacts": {
            "metrics_by_seed": "metrics_by_seed.csv",
            "metrics_aggregate": "metrics_aggregate.csv",
            "stability_pairwise": "stability_pairwise.csv",
            "stability_aggregate": "stability_aggregate.csv",
            "cross_model_agreement": "cross_model_agreement.csv",
            "cluster_count_sweep": (
                "cluster_count_sweep_by_seed.csv" if cluster_counts else None
            ),
        },
    }
    summary_path = output / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_to_serializable)
    results["summary_path"] = str(summary_path)
    if verbose:
        print(f"\nSaved complete run artifacts to: {output}")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate K-means, DEC, and IDEC on phase-boundary vPCF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["h5", "dm3", "both"], default="h5")
    parser.add_argument("--h5-file", type=str, default=str(DEFAULT_H5_FILE))
    parser.add_argument("--dm3-file", type=str, default=str(DEFAULT_DM3_FILE))
    parser.add_argument("--ground-truth-file", type=str, default=None)
    parser.add_argument("--ground-truth-dataset", type=str, default=None)
    parser.add_argument("--ground-truth-column", type=str, default=None)
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="JSON file mapping raw labels to evaluated phase IDs.",
    )
    parser.add_argument(
        "--spatial-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
    )
    parser.add_argument(
        "--model",
        choices=["kmeans", "dec", "idec", "both", "all"],
        default="all",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Required for unlabeled data; inferred from aligned ground truth otherwise.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--cluster-counts",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7],
        help="Candidate k values for the repeated K-means diagnostic sweep.",
    )
    parser.add_argument("--allow-cluster-mismatch", action="store_true")
    parser.add_argument("--ignore-label", type=int, default=None)
    parser.add_argument("--boundary-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--feature-method",
        choices=["flatten", "histogram", "statistical", "combined"],
        default="flatten",
    )
    parser.add_argument(
        "--normalize",
        choices=["minmax", "standard", "l2", "none"],
        default="minmax",
    )
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[500, 500, 2000])
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--maxiter", type=int, default=8000)
    parser.add_argument("--update-interval", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_argument_parser().parse_args()
    h5_file = args.h5_file if args.source in {"h5", "both"} else None
    dm3_file = args.dm3_file if args.source in {"dm3", "both"} else None
    normalize = None if args.normalize == "none" else args.normalize
    label_map = None
    if args.label_map is not None:
        with open(args.label_map, "r", encoding="utf-8") as handle:
            label_map = json.load(handle)

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / f"run_{args.source}_{stamp}"
    else:
        output_dir = Path(args.output_dir)

    run_pipeline(
        h5_file=h5_file,
        dm3_file=dm3_file,
        ground_truth_file=args.ground_truth_file,
        ground_truth_dataset=args.ground_truth_dataset,
        ground_truth_column=args.ground_truth_column,
        label_map=label_map,
        spatial_shape=args.spatial_shape,
        model=args.model,
        n_clusters=args.n_clusters,
        seeds=args.seeds,
        cluster_counts=args.cluster_counts,
        allow_cluster_mismatch=args.allow_cluster_mismatch,
        ignore_label=args.ignore_label,
        boundary_tolerance=args.boundary_tolerance,
        feature_method=args.feature_method,
        normalize=normalize,
        downsample_factor=args.downsample,
        max_frames=args.max_frames,
        hidden_dims=args.hidden_dims,
        pretrain_epochs=args.pretrain_epochs,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        gamma=args.gamma,
        output_dir=str(output_dir),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
