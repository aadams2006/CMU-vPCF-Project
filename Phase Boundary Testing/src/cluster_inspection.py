"""Utilities for saving cluster assignment inspection reports."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def create_cluster_report(
    labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results",
) -> pd.DataFrame:
    """Create and save a detailed sample-level cluster report."""
    os.makedirs(save_dir, exist_ok=True)

    report_data = {
        "sample_idx": np.arange(len(labels)),
        "predicted_cluster": labels,
    }

    if sample_names is not None:
        if len(sample_names) != len(labels):
            raise ValueError("sample_names length mismatch.")
        report_data["sample_name"] = sample_names

    if ground_truth_labels is not None:
        if len(ground_truth_labels) != len(labels):
            raise ValueError("ground_truth_labels length mismatch.")
        report_data["ground_truth_cluster"] = ground_truth_labels
        report_data["matches_ground_truth"] = labels == ground_truth_labels

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
) -> Dict[int, List[int]]:
    """Create and save cluster membership summaries."""
    os.makedirs(save_dir, exist_ok=True)

    n_samples = len(labels)
    unique_clusters = np.unique(labels)
    composition: Dict[int, List[int]] = {}

    for cluster_id in unique_clusters:
        sample_indices = np.where(labels == cluster_id)[0]
        if sample_names is not None:
            composition[int(cluster_id)] = list(sample_names[sample_indices])
        else:
            composition[int(cluster_id)] = list(map(int, sample_indices))

    rows = []
    for cluster_id in unique_clusters:
        samples = composition[int(cluster_id)]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": len(samples),
                "percentage": (len(samples) / n_samples) * 100.0,
                "samples": ";".join(map(str, samples)),
            }
        )

    composition_df = pd.DataFrame(rows)
    composition_path = os.path.join(save_dir, f"{model_name}_cluster_composition.csv")
    composition_df.to_csv(composition_path, index=False)
    print(f"Saved cluster composition to: {composition_path}")

    for cluster_id in unique_clusters:
        samples = composition[int(cluster_id)]
        cluster_samples_df = pd.DataFrame(
            {
                "sample_index": samples if sample_names is None else range(len(samples)),
                "sample_name": samples,
            }
        )
        cluster_file = os.path.join(save_dir, f"{model_name}_cluster_{cluster_id}_samples.csv")
        cluster_samples_df.to_csv(cluster_file, index=False)

    return composition


def create_ground_truth_comparison(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    model_name: str = "model",
    save_dir: str = "results",
) -> pd.DataFrame:
    """Create and save predicted-vs-ground-truth comparison tables."""
    os.makedirs(save_dir, exist_ok=True)

    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError("Label length mismatch.")

    matches = predicted_labels == ground_truth_labels
    accuracy = float(np.sum(matches) / len(predicted_labels))
    unique_pred = np.unique(predicted_labels)
    unique_true = np.unique(ground_truth_labels)

    comparison_rows = []
    for pred_cluster in sorted(unique_pred):
        pred_mask = predicted_labels == pred_cluster
        pred_count = int(np.sum(pred_mask))
        for true_cluster in sorted(unique_true):
            overlap = int(np.sum(pred_mask & (ground_truth_labels == true_cluster)))
            overlap_pct = (overlap / pred_count * 100.0) if pred_count else 0.0
            comparison_rows.append(
                {
                    "predicted_cluster": int(pred_cluster),
                    "ground_truth_cluster": int(true_cluster),
                    "sample_count": overlap,
                    "percentage_of_pred_cluster": overlap_pct,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(save_dir, f"{model_name}_ground_truth_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved ground truth comparison to: {comparison_path}")

    summary_path = os.path.join(save_dir, f"{model_name}_comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Cluster Prediction vs Ground Truth Comparison\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Total samples: {len(predicted_labels)}\n")
        handle.write(f"Correctly predicted: {int(np.sum(matches))} ({accuracy * 100:.2f}%)\n")
        handle.write(
            f"Incorrectly predicted: {len(predicted_labels) - int(np.sum(matches))} "
            f"({(1.0 - accuracy) * 100:.2f}%)\n\n"
        )
        handle.write(f"Predicted clusters: {len(unique_pred)}\n")
        handle.write(f"Ground truth clusters: {len(unique_true)}\n")

    return comparison_df


def create_cluster_inspection_report(
    predicted_labels: np.ndarray,
    model_name: str = "model",
    sample_names: Optional[np.ndarray] = None,
    ground_truth_labels: Optional[np.ndarray] = None,
    save_dir: str = "results",
) -> Dict[str, object]:
    """Create the full cluster inspection artifact set."""
    print(f"\nGenerating cluster inspection reports for {model_name}...")
    print(f"Save directory: {save_dir}")

    detailed_report = create_cluster_report(
        predicted_labels,
        model_name=model_name,
        sample_names=sample_names,
        ground_truth_labels=ground_truth_labels,
        save_dir=save_dir,
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
        )

    unique_clusters = np.unique(predicted_labels)
    summary_path = os.path.join(save_dir, f"{model_name}_inspection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Cluster Inspection Report for {model_name}\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Total samples: {len(predicted_labels)}\n")
        handle.write(f"Number of clusters: {len(unique_clusters)}\n\n")
        handle.write("Cluster Sizes:\n")
        for cluster_id in sorted(unique_clusters):
            size = int(np.sum(predicted_labels == cluster_id))
            pct = (size / len(predicted_labels)) * 100.0
            handle.write(f"  Cluster {int(cluster_id)}: {size} samples ({pct:.1f}%)\n")

    print(f"Saved inspection summary to: {summary_path}")
    return results

