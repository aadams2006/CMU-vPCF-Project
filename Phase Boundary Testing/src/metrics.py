"""Clustering metrics shared by DEC and IDEC models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return clustering accuracy using optimal label matching."""
    y_true = y_true.astype(np.int64)
    if y_pred.size != y_true.size:
        raise ValueError("y_true and y_pred must have the same size.")

    dim = max(int(y_pred.max()), int(y_true.max())) + 1
    weight_matrix = np.zeros((dim, dim), dtype=np.int64)
    for idx in range(y_pred.size):
        weight_matrix[y_pred[idx], y_true[idx]] += 1

    row_idx, col_idx = linear_sum_assignment(weight_matrix.max() - weight_matrix)
    matched = sum(weight_matrix[i, j] for i, j in zip(row_idx, col_idx))
    return matched / y_pred.size

