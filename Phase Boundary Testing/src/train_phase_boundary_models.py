"""Training pipeline for DEC and IDEC on phase-boundary vPCF data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_H5_FILE = PROJECT_DIR / "Data" / "SyntheticModel_HfO_80pm_vPCFs_65.h5"
DEFAULT_DM3_FILE = PROJECT_DIR / "Data" / "SyntheticModel_HfO_80pm_gaussian_HAADF.dm3"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Results"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_inspection import create_cluster_inspection_report
from metrics import acc as clustering_accuracy
from metrics import ari as adjusted_rand_index
from metrics import nmi as normalized_mutual_info
from vpcf_data_loader import (
    VPCFDataset,
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
    return value


def compute_clustering_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Optional[float]]:
    """Compute clustering metrics for a label assignment."""
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique)
    metrics: Dict[str, Optional[float]] = {
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_clusters": int(n_clusters),
        "cluster_size_min": int(counts.min()),
        "cluster_size_max": int(counts.max()),
        "cluster_size_mean": float(np.mean(counts)),
        "cluster_size_std": float(np.std(counts)),
        "cluster_imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else None,
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
        "accuracy": None,
        "normalized_mutual_info": None,
        "adjusted_rand_index": None,
    }

    if n_clusters > 1:
        try:
            metrics["silhouette_score"] = float(silhouette_score(x, labels))
        except Exception as exc:
            if verbose:
                print(f"Could not compute silhouette score: {exc}")
        try:
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(x, labels))
        except Exception as exc:
            if verbose:
                print(f"Could not compute Davies-Bouldin score: {exc}")
        try:
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(x, labels))
        except Exception as exc:
            if verbose:
                print(f"Could not compute Calinski-Harabasz score: {exc}")

    if y_true is not None:
        try:
            metrics["accuracy"] = float(clustering_accuracy(y_true, labels))
            metrics["normalized_mutual_info"] = float(normalized_mutual_info(y_true, labels))
            metrics["adjusted_rand_index"] = float(adjusted_rand_index(y_true, labels))
        except Exception as exc:
            if verbose:
                print(f"Could not compute supervised clustering metrics: {exc}")

    if verbose:
        print("\nClustering metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    return metrics


def get_model_dimensions(input_dim: int, n_clusters: int, hidden_dims: Optional[list] = None) -> list:
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
    verbose: bool = True,
) -> Tuple[object, np.ndarray]:
    """Train a DEC model."""
    from dec import DEC

    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Training DEC")
        print("=" * 60)
        print(f"Input dimension: {x.shape[1]}")
        print(f"Samples: {x.shape[0]}")
        print(f"Clusters: {n_clusters}")
        print(f"Architecture: {dims}")
        print(f"Output directory: {save_dir}")

    model = DEC(dims=dims, n_clusters=n_clusters, save_dir=save_dir)
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
    verbose: bool = True,
) -> Tuple[object, np.ndarray]:
    """Train an IDEC model."""
    from idec import IDEC

    dims = get_model_dimensions(x.shape[1], n_clusters, hidden_dims)
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Training IDEC")
        print("=" * 60)
        print(f"Input dimension: {x.shape[1]}")
        print(f"Samples: {x.shape[0]}")
        print(f"Clusters: {n_clusters}")
        print(f"Architecture: {dims}")
        print(f"Gamma: {gamma}")
        print(f"Output directory: {save_dir}")

    model = IDEC(dims=dims, n_clusters=n_clusters, gamma=gamma, save_dir=save_dir)
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


def save_results(
    labels: np.ndarray,
    dataset: VPCFDataset,
    save_dir: str,
    model_name: str,
    model,
    metrics: Dict[str, Optional[float]],
    training_config: Dict[str, object],
) -> None:
    """Save model outputs, metrics, and inspection artifacts."""
    os.makedirs(save_dir, exist_ok=True)

    labels_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(labels)),
            "sample_name": dataset.sample_names,
            "cluster": labels,
        }
    )
    labels_path = os.path.join(save_dir, f"{model_name}_cluster_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved cluster labels to: {labels_path}")

    unique, counts = np.unique(labels, return_counts=True)
    stats_df = pd.DataFrame(
        {
            "cluster": unique,
            "count": counts,
            "percentage": counts / len(labels) * 100.0,
        }
    )
    stats_path = os.path.join(save_dir, f"{model_name}_cluster_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved cluster statistics to: {stats_path}")

    centers = model.get_cluster_centers()
    centers_df = pd.DataFrame(centers, columns=[f"dim_{idx}" for idx in range(centers.shape[1])])
    centers_df.insert(0, "cluster", np.arange(centers.shape[0]))
    centers_path = os.path.join(save_dir, f"{model_name}_cluster_centers.csv")
    centers_df.to_csv(centers_path, index=False)
    print(f"Saved cluster centers to: {centers_path}")

    latent = model.extract_features(dataset.features)
    latent_df = pd.DataFrame(latent, columns=[f"latent_dim_{idx}" for idx in range(latent.shape[1])])
    latent_df.insert(0, "cluster", labels)
    latent_df.insert(0, "sample_name", dataset.sample_names)
    latent_df.insert(0, "sample_idx", np.arange(len(labels)))
    latent_path = os.path.join(save_dir, f"{model_name}_latent_features.csv")
    latent_df.to_csv(latent_path, index=False)
    print(f"Saved latent features to: {latent_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    metadata = {
        "model_name": model_name,
        "source_file": dataset.source_file,
        "n_samples": len(labels),
        "training_config": training_config,
        "metrics": metrics,
    }
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_to_serializable)
    print(f"Saved metadata to: {metadata_path}")

    create_cluster_inspection_report(
        predicted_labels=labels,
        model_name=model_name,
        sample_names=dataset.sample_names,
        ground_truth_labels=dataset.labels,
        save_dir=save_dir,
    )


def run_pipeline(
    h5_file: Optional[str] = None,
    dm3_file: Optional[str] = None,
    model: str = "both",
    n_clusters: int = 3,
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
    """Run the complete phase-boundary training pipeline."""
    if h5_file is None and dm3_file is None:
        raise ValueError("At least one of h5_file or dm3_file must be provided.")

    dependencies = check_dependencies()
    if verbose:
        print("Checking dependencies:")
        for name, available in dependencies.items():
            mark = "[YES]" if available else "[NO]"
            print(f"  {mark} {name}")
        print()

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
    if verbose:
        print(f"\nFinal dataset: {dataset}")
        print(f"Feature matrix shape: {dataset.features.shape}")

    x = dataset.features
    y = dataset.labels

    training_config = {
        "h5_file": h5_file,
        "dm3_file": dm3_file,
        "model": model,
        "n_clusters": n_clusters,
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

    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "dataset": {
            "source_file": dataset.source_file,
            "n_samples": dataset.n_samples,
            "feature_dim": dataset.feature_dim,
        },
        "training_config": training_config,
        "models": {},
    }
    results: Dict[str, object] = {"dataset": dataset}

    if model in {"dec", "both"}:
        dec_dir = os.path.join(output_dir, "dec")
        dec_model, dec_labels = train_dec_model(
            x=x,
            y=y,
            n_clusters=n_clusters,
            save_dir=dec_dir,
            hidden_dims=hidden_dims,
            pretrain_epochs=pretrain_epochs,
            maxiter=maxiter,
            update_interval=update_interval,
            batch_size=batch_size,
            verbose=verbose,
        )
        dec_metrics = compute_clustering_metrics(x, dec_labels, y_true=y, verbose=verbose)
        save_results(dec_labels, dataset, dec_dir, "dec", dec_model, dec_metrics, training_config)
        results.update({"dec_model": dec_model, "dec_labels": dec_labels, "dec_metrics": dec_metrics})
        summary["models"]["dec"] = dec_metrics

    if model in {"idec", "both"}:
        idec_dir = os.path.join(output_dir, "idec")
        idec_model, idec_labels = train_idec_model(
            x=x,
            y=y,
            n_clusters=n_clusters,
            save_dir=idec_dir,
            hidden_dims=hidden_dims,
            pretrain_epochs=pretrain_epochs,
            maxiter=maxiter,
            update_interval=update_interval,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose,
        )
        idec_metrics = compute_clustering_metrics(x, idec_labels, y_true=y, verbose=verbose)
        save_results(idec_labels, dataset, idec_dir, "idec", idec_model, idec_metrics, training_config)
        results.update({"idec_model": idec_model, "idec_labels": idec_labels, "idec_metrics": idec_metrics})
        summary["models"]["idec"] = idec_metrics

    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_to_serializable)
    if verbose:
        print(f"\nSaved run summary to: {summary_path}")
        print(f"Results saved to: {output_dir}")

    results["summary_path"] = summary_path
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train DEC/IDEC models on the phase-boundary vPCF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["h5", "dm3", "both"],
        default="h5",
        help="Which data source(s) to train on.",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default=str(DEFAULT_H5_FILE),
        help="Path to the HDF5 vPCF file.",
    )
    parser.add_argument(
        "--dm3-file",
        type=str,
        default=str(DEFAULT_DM3_FILE),
        help="Path to the DM3/DM4 microscopy file.",
    )
    parser.add_argument(
        "--model",
        choices=["dec", "idec", "both"],
        default="both",
        help="Which model(s) to train.",
    )
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters.")
    parser.add_argument(
        "--feature-method",
        choices=["flatten", "histogram", "statistical", "combined"],
        default="flatten",
        help="Feature extraction method.",
    )
    parser.add_argument(
        "--normalize",
        choices=["minmax", "standard", "l2", "none"],
        default="minmax",
        help="Feature normalization method.",
    )
    parser.add_argument("--downsample", type=int, default=None, help="Image downsampling factor.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of H5 frames to load.")
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[500, 500, 2000],
        help="Hidden layer dimensions.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="Autoencoder pretraining epochs.")
    parser.add_argument("--maxiter", type=int, default=8000, help="Maximum clustering iterations.")
    parser.add_argument("--update-interval", type=int, default=140, help="Clustering update interval.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--gamma", type=float, default=0.1, help="IDEC reconstruction loss weight.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where run artifacts will be saved. Defaults to Results/<source>_only.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")
    return parser


def main():
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    h5_file = args.h5_file if args.source in {"h5", "both"} else None
    dm3_file = args.dm3_file if args.source in {"dm3", "both"} else None
    normalize = None if args.normalize == "none" else args.normalize

    if args.output_dir is None:
        if args.source == "both":
            output_dir = DEFAULT_OUTPUT_DIR / "combined"
        elif args.source == "dm3":
            output_dir = DEFAULT_OUTPUT_DIR / "dm3_only"
        else:
            output_dir = DEFAULT_OUTPUT_DIR / "h5_only"
    else:
        output_dir = Path(args.output_dir)

    run_pipeline(
        h5_file=h5_file,
        dm3_file=dm3_file,
        model=args.model,
        n_clusters=args.n_clusters,
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
