"""Pipeline to evaluate DEC and IDEC on the real vPCF datasets."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

import metrics
from DEC import DEC
from IDEC import IDEC


def resolve_data_paths(patterns: Sequence[str]) -> List[str]:
    """Expand glob patterns into a sorted list of dataset paths."""

    resolved: List[str] = []
    for pattern in patterns:
        resolved.extend(glob.glob(pattern))
    unique_paths = sorted(set(resolved))
    if not unique_paths:
        raise FileNotFoundError(
            "No vPCF JSON files were found. Check the --data-glob argument."
        )
    return unique_paths


def load_vpcf_dataset(
    paths: Iterable[str],
    limit_per_label: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the vPCF arrays and associated labels from JSON files."""

    features: List[List[float]] = []
    labels: List[str] = []
    per_label_counts: defaultdict[str, int] = defaultdict(int)
    feature_lengths: set[int] = set()

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for label, content in payload.items():
            rows = content.get("$array", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if limit_per_label is not None and per_label_counts[label] >= limit_per_label:
                    break
                if not isinstance(row, list):
                    continue
                feature_lengths.add(len(row))
                features.append(row)
                labels.append(label)
                per_label_counts[label] += 1

    if not features:
        raise ValueError("The provided vPCF files did not yield any feature vectors.")
    if len(feature_lengths) > 1:
        raise ValueError(
            f"Inconsistent feature lengths detected: {sorted(feature_lengths)}"
        )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    return x, y


def write_label_mapping(path: os.PathLike[str] | str, encoder: LabelEncoder) -> None:
    """Persist a CSV with the mapping between encoded and original labels."""

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["encoded", "label"])
        for encoded, label in enumerate(encoder.classes_):
            writer.writerow([encoded, label])


def summarize_dataset(x: np.ndarray, y: np.ndarray) -> None:
    """Print dataset statistics to stdout."""

    print("--- vPCF Dataset Summary ---")
    print(f"Total samples: {x.shape[0]}")
    print(f"Feature dimension: {x.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Number of unique labels: {unique.size}")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    print("-" * 30)


def compute_final_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a consistent set of clustering metrics."""

    return {
        "acc": float(metrics.acc(y_true, y_pred)),
        "nmi": float(metrics.nmi(y_true, y_pred)),
        "ari": float(metrics.ari(y_true, y_pred)),
    }


def save_metric_report(path: os.PathLike[str] | str, values: dict[str, float]) -> None:
    """Write metric values to CSV."""

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in values.items():
            writer.writerow([key, value])


def run_dec_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    dims: List[int],
    save_dir: os.PathLike[str] | str,
    batch_size: int,
    pretrain_epochs: int,
    maxiter: int,
    update_interval: int,
) -> np.ndarray:
    """Train DEC on the provided dataset and persist outputs."""

    os.makedirs(save_dir, exist_ok=True)
    dec = DEC(dims=dims, n_clusters=dims[-1], save_dir=str(save_dir))
    dec.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    dec.compile(optimizer="sgd")
    labels = dec.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size,
    )
    metrics_path = Path(save_dir) / "final_metrics.csv"
    save_metric_report(metrics_path, compute_final_metrics(y, labels))
    return labels


def run_idec_pipeline(
    x: np.ndarray,
    y: np.ndarray,
    dims: List[int],
    save_dir: os.PathLike[str] | str,
    batch_size: int,
    pretrain_epochs: int,
    maxiter: int,
    update_interval: int,
) -> np.ndarray:
    """Train IDEC on the provided dataset and persist outputs."""

    os.makedirs(save_dir, exist_ok=True)
    idec = IDEC(dims=dims, n_clusters=dims[-1], save_dir=str(save_dir))
    if not idec.pretrained:
        idec.pretrain(x, epochs=pretrain_epochs, batch_size=batch_size)
    idec.compile(optimizer="sgd")
    labels = idec.fit(
        x,
        y=y,
        maxiter=maxiter,
        update_interval=update_interval,
        batch_size=batch_size,
    )
    metrics_path = Path(save_dir) / "final_metrics.csv"
    save_metric_report(metrics_path, compute_final_metrics(y, labels))
    return labels


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DEC and IDEC autoencoders on the real vPCF datasets.",
    )
    parser.add_argument(
        "--data-glob",
        nargs="+",
        default=["data/saved_vpcfs/*.json"],
        help="Glob patterns pointing to JSON files with vPCF arrays.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where model artifacts and reports will be stored.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[500, 500, 2000],
        help="Hidden layer sizes for the autoencoder backbone (excluding input and cluster layers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for both pretraining and fine-tuning phases.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Number of epochs to use during the autoencoder pretraining stage.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=8000,
        help="Maximum number of clustering iterations for DEC/IDEC.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=200,
        help="Number of iterations between target distribution updates.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=None,
        help="Optional cap on the number of samples to load per label.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_argument_parser()
    if args is None:
        args = parser.parse_args()
    else:
        # Support programmatic usage where ``args`` is already populated.
        if not isinstance(args, argparse.Namespace):
            raise TypeError("args must be an argparse.Namespace when provided explicitly.")
        args = parser.parse_args([], namespace=args)

    data_paths = resolve_data_paths(args.data_glob)
    x, labels = load_vpcf_dataset(data_paths, limit_per_label=args.limit_per_label)
    summarize_dataset(x, labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping_path = output_dir / "vpcf_label_mapping.csv"
    write_label_mapping(label_mapping_path, encoder)
    print(f"Label mapping saved to {label_mapping_path}")

    hidden_dims = list(args.hidden_dims)
    if not hidden_dims:
        raise ValueError("At least one hidden dimension must be specified.")
    dims = [x.shape[1], *hidden_dims, len(encoder.classes_)]

    update_interval = max(1, min(args.update_interval, x.shape[0]))

    dec_dir = output_dir / "dec" / "vpcf"
    run_dec_pipeline(
        x=x,
        y=y,
        dims=dims,
        save_dir=dec_dir,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        maxiter=args.maxiter,
        update_interval=update_interval,
    )
    print(f"DEC final metrics saved to {dec_dir / 'final_metrics.csv'}")

    idec_dir = output_dir / "idec" / "vpcf"
    run_idec_pipeline(
        x=x,
        y=y,
        dims=dims,
        save_dir=idec_dir,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        maxiter=args.maxiter,
        update_interval=update_interval,
    )
    print(f"IDEC final metrics saved to {idec_dir / 'final_metrics.csv'}")


if __name__ == "main__":  # pragma: no cover - defensive typo check
    raise RuntimeError("This module should be executed as a script via '__main__'.")


if __name__ == "__main__":
    main()
