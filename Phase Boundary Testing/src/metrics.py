"""Evaluation utilities shared by phase-boundary clustering workflows.

Cluster identifiers are arbitrary. Every label-sensitive metric in this module
therefore aligns predicted cluster IDs to reference labels before it compares
them. Label-invariant metrics such as ARI and NMI use the raw assignments.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def _as_1d_labels(labels: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return values


def valid_label_mask(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    ignore_label: Optional[int] = None,
) -> np.ndarray:
    """Return the samples eligible for supervised evaluation."""
    truth = _as_1d_labels(y_true, "y_true")
    mask = np.ones(truth.size, dtype=bool)
    if ignore_label is not None:
        mask &= truth != ignore_label
    if np.issubdtype(truth.dtype, np.floating):
        mask &= np.isfinite(truth)

    if y_pred is not None:
        pred = _as_1d_labels(y_pred, "y_pred")
        if pred.size != truth.size:
            raise ValueError("y_true and y_pred must have the same size.")
        if np.issubdtype(pred.dtype, np.floating):
            mask &= np.isfinite(pred)
    return mask


def optimal_label_mapping(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: Optional[int] = None,
) -> Dict[int, int]:
    """Map predicted IDs to reference IDs using maximum-overlap assignment."""
    truth = _as_1d_labels(y_true, "y_true")
    pred = _as_1d_labels(y_pred, "y_pred")
    mask = valid_label_mask(truth, pred, ignore_label=ignore_label)
    if not np.any(mask):
        raise ValueError("No valid ground-truth samples remain after filtering.")

    truth_valid = truth[mask]
    pred_valid = pred[mask]
    true_values = np.unique(truth_valid)
    pred_values = np.unique(pred_valid)
    contingency = np.zeros((pred_values.size, true_values.size), dtype=np.int64)

    for pred_idx, pred_value in enumerate(pred_values):
        pred_mask = pred_valid == pred_value
        for true_idx, true_value in enumerate(true_values):
            contingency[pred_idx, true_idx] = int(
                np.sum(pred_mask & (truth_valid == true_value))
            )

    row_idx, col_idx = linear_sum_assignment(contingency.max() - contingency)
    mapping = {
        int(pred_values[pred_idx]): int(true_values[true_idx])
        for pred_idx, true_idx in zip(row_idx, col_idx)
    }

    # Preserve every predicted cluster if k exceeds the reference class count.
    for pred_idx, pred_value in enumerate(pred_values):
        pred_key = int(pred_value)
        if pred_key not in mapping:
            mapping[pred_key] = int(true_values[np.argmax(contingency[pred_idx])])
    return mapping


def remap_labels(
    labels: np.ndarray,
    mapping: Mapping[int, int],
    unmapped_value: int = -1,
) -> np.ndarray:
    """Apply a predicted-to-reference cluster mapping."""
    values = _as_1d_labels(labels, "labels")
    return np.asarray(
        [mapping.get(int(value), unmapped_value) for value in values],
        dtype=np.int64,
    )


def acc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: Optional[int] = None,
) -> float:
    """Return clustering accuracy after optimal label matching."""
    truth = _as_1d_labels(y_true, "y_true")
    pred = _as_1d_labels(y_pred, "y_pred")
    mask = valid_label_mask(truth, pred, ignore_label=ignore_label)
    mapping = optimal_label_mapping(truth, pred, ignore_label=ignore_label)
    matched = remap_labels(pred, mapping)
    return float(np.mean(matched[mask] == truth[mask]))


def compute_internal_metrics(x: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Compute unsupervised metrics in one common feature space."""
    features = np.asarray(x)
    pred = _as_1d_labels(labels, "labels")
    if features.ndim != 2 or features.shape[0] != pred.size:
        raise ValueError("x must be a 2-D matrix with one row per label.")

    unique, counts = np.unique(pred, return_counts=True)
    result: Dict[str, object] = {
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_clusters": int(unique.size),
        "cluster_size_min": int(counts.min()),
        "cluster_size_max": int(counts.max()),
        "cluster_size_mean": float(np.mean(counts)),
        "cluster_size_std": float(np.std(counts)),
        "cluster_imbalance_ratio": (
            float(counts.max() / counts.min()) if counts.min() > 0 else None
        ),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }

    if 1 < unique.size < features.shape[0]:
        for key, scorer in (
            ("silhouette_score", silhouette_score),
            ("davies_bouldin_score", davies_bouldin_score),
            ("calinski_harabasz_score", calinski_harabasz_score),
        ):
            try:
                result[key] = float(scorer(features, pred))
            except (TypeError, ValueError, FloatingPointError):
                result[key] = None
    return result


def compute_external_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: Optional[int] = None,
) -> Tuple[Dict[str, object], Dict[int, int]]:
    """Compute ground-truth metrics and return the optimal ID mapping."""
    truth = _as_1d_labels(y_true, "y_true")
    pred = _as_1d_labels(y_pred, "y_pred")
    mask = valid_label_mask(truth, pred, ignore_label=ignore_label)
    if not np.any(mask):
        raise ValueError("No valid ground-truth samples remain after filtering.")

    mapping = optimal_label_mapping(truth, pred, ignore_label=ignore_label)
    matched = remap_labels(pred, mapping)
    truth_valid = truth[mask].astype(np.int64)
    pred_valid = pred[mask].astype(np.int64)
    matched_valid = matched[mask]
    result: Dict[str, object] = {
        "n_labeled_samples": int(mask.sum()),
        "matched_accuracy": float(np.mean(matched_valid == truth_valid)),
        "normalized_mutual_info": float(nmi(truth_valid, pred_valid)),
        "adjusted_rand_index": float(ari(truth_valid, pred_valid)),
        "macro_f1": float(f1_score(truth_valid, matched_valid, average="macro")),
        "weighted_f1": float(f1_score(truth_valid, matched_valid, average="weighted")),
        "label_mapping": {str(key): int(value) for key, value in mapping.items()},
    }
    return result, mapping


def confusion_matrix_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return class IDs and a matched-label confusion matrix."""
    truth = _as_1d_labels(y_true, "y_true")
    pred = _as_1d_labels(y_pred, "y_pred")
    mask = valid_label_mask(truth, pred, ignore_label=ignore_label)
    mapping = optimal_label_mapping(truth, pred, ignore_label=ignore_label)
    matched = remap_labels(pred, mapping)
    classes = np.unique(truth[mask]).astype(np.int64)
    matrix = np.zeros((classes.size, classes.size), dtype=np.int64)
    class_to_idx = {int(value): idx for idx, value in enumerate(classes)}
    for true_value, pred_value in zip(truth[mask], matched[mask]):
        pred_idx = class_to_idx.get(int(pred_value))
        if pred_idx is not None:
            matrix[class_to_idx[int(true_value)], pred_idx] += 1
    return classes, matrix


def boundary_map(labels: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a two-sided four-neighbour boundary map from a label grid."""
    grid = np.asarray(labels)
    if grid.ndim != 2:
        raise ValueError("labels must be a 2-D grid for boundary evaluation.")
    valid = np.ones(grid.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != grid.shape:
        raise ValueError("valid_mask must have the same shape as labels.")

    result = np.zeros(grid.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (grid[:, 1:] != grid[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (grid[1:, :] != grid[:-1, :])
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def _safe_ratio(numerator: float, denominator: float, both_empty: float = 1.0) -> float:
    if denominator == 0:
        return both_empty
    return float(numerator / denominator)


def compute_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spatial_shape: Sequence[int],
    tolerance: float = 1.0,
    ignore_label: Optional[int] = None,
) -> Dict[str, object]:
    """Compute spatial phase-boundary overlap and distance metrics."""
    if len(spatial_shape) != 2:
        raise ValueError("spatial_shape must contain exactly (height, width).")
    shape = (int(spatial_shape[0]), int(spatial_shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("spatial_shape values must be positive.")

    truth = _as_1d_labels(y_true, "y_true")
    pred = _as_1d_labels(y_pred, "y_pred")
    if truth.size != shape[0] * shape[1] or pred.size != truth.size:
        raise ValueError("spatial_shape must contain exactly one cell per label.")

    valid_flat = valid_label_mask(truth, pred, ignore_label=ignore_label)
    mapping = optimal_label_mapping(truth, pred, ignore_label=ignore_label)
    matched = remap_labels(pred, mapping)
    truth_grid = truth.reshape(shape)
    pred_grid = matched.reshape(shape)
    valid_grid = valid_flat.reshape(shape)
    truth_boundary = boundary_map(truth_grid, valid_grid)
    pred_boundary = boundary_map(pred_grid, valid_grid)

    intersection = int(np.sum(truth_boundary & pred_boundary))
    union = int(np.sum(truth_boundary | pred_boundary))
    true_count = int(truth_boundary.sum())
    pred_count = int(pred_boundary.sum())
    exact_iou = _safe_ratio(intersection, union)
    exact_dice = _safe_ratio(2.0 * intersection, true_count + pred_count)

    if true_count == 0 and pred_count == 0:
        precision = recall = tolerant_f1 = 1.0
        mean_distance = percentile_95 = 0.0
    elif true_count == 0 or pred_count == 0:
        precision = recall = tolerant_f1 = 0.0
        mean_distance = percentile_95 = None
    else:
        distance_to_truth = distance_transform_edt(~truth_boundary)
        distance_to_pred = distance_transform_edt(~pred_boundary)
        pred_distances = distance_to_truth[pred_boundary]
        true_distances = distance_to_pred[truth_boundary]
        precision = float(np.mean(pred_distances <= tolerance))
        recall = float(np.mean(true_distances <= tolerance))
        tolerant_f1 = _safe_ratio(
            2.0 * precision * recall,
            precision + recall,
            both_empty=0.0,
        )
        symmetric = np.concatenate([pred_distances, true_distances])
        mean_distance = float(np.mean(symmetric))
        percentile_95 = float(np.percentile(symmetric, 95))

    return {
        "boundary_tolerance_px": float(tolerance),
        "true_boundary_pixels": true_count,
        "predicted_boundary_pixels": pred_count,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": tolerant_f1,
        "boundary_iou_exact": exact_iou,
        "boundary_dice_exact": exact_dice,
        "mean_symmetric_boundary_distance_px": mean_distance,
        "boundary_distance_p95_px": percentile_95,
    }


def evaluate_clustering(
    x: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    boundary_tolerance: float = 1.0,
    ignore_label: Optional[int] = None,
) -> Dict[str, object]:
    """Compute every applicable internal, external, and boundary metric."""
    result = compute_internal_metrics(x, labels)
    result.update(
        {
            "n_labeled_samples": None,
            "matched_accuracy": None,
            "normalized_mutual_info": None,
            "adjusted_rand_index": None,
            "macro_f1": None,
            "weighted_f1": None,
            "label_mapping": None,
        }
    )
    if y_true is not None:
        external, _ = compute_external_metrics(y_true, labels, ignore_label=ignore_label)
        result.update(external)
        if spatial_shape is not None:
            result.update(
                compute_boundary_metrics(
                    y_true,
                    labels,
                    spatial_shape=spatial_shape,
                    tolerance=boundary_tolerance,
                    ignore_label=ignore_label,
                )
            )
    return result


def pairwise_stability(label_runs: Mapping[int, np.ndarray]) -> list[Dict[str, object]]:
    """Compute label-invariant agreement for every pair of random seeds."""
    rows: list[Dict[str, object]] = []
    for seed_a, seed_b in combinations(sorted(label_runs), 2):
        labels_a = _as_1d_labels(label_runs[seed_a], f"labels[{seed_a}]")
        labels_b = _as_1d_labels(label_runs[seed_b], f"labels[{seed_b}]")
        if labels_a.size != labels_b.size:
            raise ValueError("All stability runs must contain the same samples.")
        rows.append(
            {
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "adjusted_rand_index": float(ari(labels_a, labels_b)),
                "normalized_mutual_info": float(nmi(labels_a, labels_b)),
            }
        )
    return rows


def aggregate_numeric_metrics(
    records: Iterable[Mapping[str, object]],
    confidence: float = 0.95,
    excluded_keys: Iterable[str] = ("seed", "n_samples", "n_features", "n_clusters"),
) -> list[Dict[str, object]]:
    """Return mean, sample standard deviation, and t confidence intervals."""
    rows = list(records)
    if not rows:
        return []
    excluded = set(excluded_keys)
    keys = sorted({key for row in rows for key in row if key not in excluded})
    summary: list[Dict[str, object]] = []

    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                continue
            value_float = float(value)
            if np.isfinite(value_float):
                values.append(value_float)
        if not values:
            continue

        values_array = np.asarray(values, dtype=float)
        n_values = int(values_array.size)
        mean = float(values_array.mean())
        if n_values > 1:
            std = float(values_array.std(ddof=1))
            critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=n_values - 1))
            half_width = critical * std / np.sqrt(n_values)
        else:
            std = 0.0
            half_width = 0.0
        summary.append(
            {
                "metric": key,
                "n": n_values,
                "mean": mean,
                "std": std,
                "confidence": float(confidence),
                "ci_low": mean - half_width,
                "ci_high": mean + half_width,
            }
        )
    return summary
