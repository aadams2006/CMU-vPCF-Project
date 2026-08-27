"""Utilities for saving auditable cluster-assignment reports."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .metrics import confusion_matrix_table, optimal_label_mapping, remap_labels
except ImportError:
    from metrics import confusion_matrix_table, optimal_label_mapping, remap_labels


def create_cluster_report(
    labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results",
    ignore_label: Optional[int] = None,
) -> pd.DataFrame:
    """Create a sample-level report while preserving raw cluster IDs."""
    os.makedirs(save_dir, exist_ok=True)
    labels = np.asarray(labels).reshape(-1)
    report_data = {
        "sample_idx": np.arange(len(labels)),
        "predicted_cluster_raw": labels,
    }

    if sample_names is not None:
        if len(sample_names) != len(labels):
            raise ValueError("sample_names length mismatch.")
        report_data["sample_name"] = sample_names

    if ground_truth_labels is not None:
        truth = np.asarray(ground_truth_labels).reshape(-1)
        if len(truth) != len(labels):
            raise ValueError("ground_truth_labels length mismatch.")
        mapping = optimal_label_mapping(truth, labels, ignore_label=ignore_label)
        matched = remap_labels(labels, mapping)
        valid = np.ones(len(truth), dtype=bool)
        if ignore_label is not None:
            valid &= truth != ignore_label
        report_data["predicted_cluster_matched"] = matched
        report_data["ground_truth_cluster"] = truth
        report_data["is_evaluated"] = valid
        report_data["matches_ground_truth"] = valid & (matched == truth)

    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(save_dir, f"{model_name}_detailed_cluster_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Saved detailed cluster report to: {report_path}")
    return report_df


def create_cluster_composition(
    labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    save_dir: str = "results",
) -> Dict[int, List[object]]:
    """Create and save cluster membership summaries."""
    os.makedirs(save_dir, exist_ok=True)
    labels = np.asarray(labels).reshape(-1)
    if sample_names is not None and len(sample_names) != len(labels):
        raise ValueError("sample_names length mismatch.")

    n_samples = len(labels)
    unique_clusters = np.unique(labels)
    composition: Dict[int, List[object]] = {}
    rows = []

    for cluster_id in unique_clusters:
        sample_indices = np.where(labels == cluster_id)[0]
        samples = (
            list(np.asarray(sample_names)[sample_indices])
            if sample_names is not None
            else list(map(int, sample_indices))
        )
        composition[int(cluster_id)] = samples
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": len(samples),
                "percentage": (len(samples) / n_samples) * 100.0,
                "samples": ";".join(map(str, samples)),
            }
        )
        pd.DataFrame(
            {
                "sample_idx": sample_indices,
                "sample_name": (
                    np.asarray(sample_names)[sample_indices]
                    if sample_names is not None
                    else sample_indices
                ),
            }
        ).to_csv(
            os.path.join(save_dir, f"{model_name}_cluster_{int(cluster_id)}_samples.csv"),
            index=False,
        )

    composition_path = os.path.join(save_dir, f"{model_name}_cluster_composition.csv")
    pd.DataFrame(rows).to_csv(composition_path, index=False)
    print(f"Saved cluster composition to: {composition_path}")
    return composition


def create_ground_truth_comparison(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    model_name: str = "model",
    save_dir: str = "results",
    ignore_label: Optional[int] = None,
) -> pd.DataFrame:
    """Save mapping, confusion, and matched predicted-vs-reference tables."""
    os.makedirs(save_dir, exist_ok=True)
    pred = np.asarray(predicted_labels).reshape(-1)
    truth = np.asarray(ground_truth_labels).reshape(-1)
    if len(pred) != len(truth):
        raise ValueError("Label length mismatch.")

    mapping = optimal_label_mapping(truth, pred, ignore_label=ignore_label)
    matched = remap_labels(pred, mapping)
    valid = np.ones(len(truth), dtype=bool)
    if ignore_label is not None:
        valid &= truth != ignore_label
    matches = valid & (matched == truth)

    mapping_path = os.path.join(save_dir, f"{model_name}_label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as handle:
        json.dump({str(key): int(value) for key, value in mapping.items()}, handle, indent=2)

    unique_pred = np.unique(pred)
    unique_true = np.unique(truth[valid])
    comparison_rows = []
    for pred_cluster in sorted(unique_pred):
        pred_mask = valid & (pred == pred_cluster)
        pred_count = int(np.sum(pred_mask))
        for true_cluster in sorted(unique_true):
            overlap = int(np.sum(pred_mask & (truth == true_cluster)))
            comparison_rows.append(
                {
                    "predicted_cluster_raw": int(pred_cluster),
                    "mapped_ground_truth_cluster": int(mapping[int(pred_cluster)]),
                    "ground_truth_cluster": int(true_cluster),
                    "sample_count": overlap,
                    "percentage_of_predicted_cluster": (
                        overlap / pred_count * 100.0 if pred_count else 0.0
                    ),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        os.path.join(save_dir, f"{model_name}_ground_truth_comparison.csv"),
        index=False,
    )

    classes, matrix = confusion_matrix_table(truth, pred, ignore_label=ignore_label)
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"true_{int(value)}" for value in classes],
        columns=[f"predicted_{int(value)}" for value in classes],
    )
    confusion_df.index.name = "ground_truth"
    confusion_df.to_csv(os.path.join(save_dir, f"{model_name}_confusion_matrix.csv"))

    evaluated = int(valid.sum())
    correct = int(matches.sum())
    summary_path = os.path.join(save_dir, f"{model_name}_comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Cluster Prediction vs Ground Truth Comparison\n")
        handle.write("=" * 50 + "\n\n")
        handle.write("Cluster IDs were aligned with maximum-overlap Hungarian matching.\n")
        handle.write(f"Total samples: {len(pred)}\n")
        handle.write(f"Evaluated samples: {evaluated}\n")
        handle.write(f"Correct after matching: {correct} ({correct / evaluated * 100:.2f}%)\n")
        handle.write(f"Incorrect after matching: {evaluated - correct}\n")
        handle.write(f"Predicted clusters: {len(unique_pred)}\n")
        handle.write(f"Ground-truth clusters: {len(unique_true)}\n")
    return comparison_df


def create_cluster_inspection_report(
    predicted_labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results",
    ignore_label: Optional[int] = None,
) -> Dict[str, object]:
    """Create the complete cluster-inspection artifact set."""
    print(f"\nGenerating cluster inspection reports for {model_name}...")
    detailed_report = create_cluster_report(
        predicted_labels,
        model_name=model_name,
        sample_names=sample_names,
        ground_truth_labels=ground_truth_labels,
        save_dir=save_dir,
        ignore_label=ignore_label,
    )
    composition = create_cluster_composition(
        predicted_labels,
        model_name=model_name,
        sample_names=sample_names,
        save_dir=save_dir,
    )
    results: Dict[str, object] = {
        "detailed_report": detailed_report,
        "composition": composition,
    }
    if ground_truth_labels is not None:
        results["comparison"] = create_ground_truth_comparison(
            predicted_labels,
            ground_truth_labels,
            model_name=model_name,
            save_dir=save_dir,
            ignore_label=ignore_label,
        )

    labels = np.asarray(predicted_labels).reshape(-1)
    unique_clusters = np.unique(labels)
    summary_path = os.path.join(save_dir, f"{model_name}_inspection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Cluster Inspection Report for {model_name}\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Total samples: {len(labels)}\n")
        handle.write(f"Number of clusters: {len(unique_clusters)}\n\n")
        handle.write("Raw Cluster Sizes:\n")
        for cluster_id in sorted(unique_clusters):
            size = int(np.sum(labels == cluster_id))
            handle.write(
                f"  Cluster {int(cluster_id)}: {size} samples "
                f"({size / len(labels) * 100:.1f}%)\n"
            )
    print(f"Saved inspection summary to: {summary_path}")
    return results
