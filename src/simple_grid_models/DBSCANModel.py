"""DBSCAN clustering utilities for the project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .. import metrics
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
)
import csv
import os


@dataclass
class DBSCANClustering:
    """Convenience wrapper around :class:`sklearn.cluster.DBSCAN`.

    Parameters
    ----------
    eps: float, default=0.5
        The maximum distance between two samples for one to be considered as in the
        neighborhood of the other. Passed directly to ``DBSCAN``.
    min_samples: int, default=5
        The number of samples in a neighborhood for a point to be considered as a
        core point. Passed directly to ``DBSCAN``.
    metric: str, default="euclidean"
        Metric used by ``DBSCAN``.
    scale: bool, default=True
        Whether to apply :class:`~sklearn.preprocessing.StandardScaler` before
        clustering. Scaling often improves DBSCAN performance on datasets with
        heterogeneous feature scales.
    treat_noise_as_cluster: bool, default=False
        Whether to convert all noise points (label ``-1``) to their own dedicated
        cluster when returning labels. The evaluation step always ignores noise, so
        this option is mainly useful for downstream analysis that requires a label
        for every sample.
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    scale: bool = True
    treat_noise_as_cluster: bool = False

    def __post_init__(self) -> None:
        self._estimator: Optional[DBSCAN] = None
        self._scaler: Optional[StandardScaler] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> np.ndarray:
        """Cluster the provided dataset.

        Parameters
        ----------
        x: numpy.ndarray
            Feature matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Cluster labels for each sample. Noise points are marked with ``-1``
            unless ``treat_noise_as_cluster`` is set to ``True``.
        """

        if x.ndim != 2:
            raise ValueError("DBSCANClustering expects a 2D array as input.")

        processed = x
        if self.scale:
            self._scaler = StandardScaler(with_mean=True, with_std=True)
            processed = self._scaler.fit_transform(x)

        self._estimator = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        raw_labels = self._estimator.fit_predict(processed)
        self.labels_ = self._handle_noise(raw_labels)
        return self.labels_

    def _handle_noise(self, labels: np.ndarray) -> np.ndarray:
        if not self.treat_noise_as_cluster:
            return labels
        if labels.size == 0:
            return labels
        result = labels.copy()
        if np.any(result == -1):
            non_noise = result[result != -1]
            offset = non_noise.max(initial=-1) + 1
            result[result == -1] = offset
        return result

    def evaluate(self, y_true: np.ndarray) -> Dict[str, float]:
        """Compute clustering metrics on the fitted labels.

        Noise points (labelled ``-1``) are excluded from the comparison because
        most clustering metrics expect finite cluster ids.
        """

        if self.labels_ is None:
            raise RuntimeError("Call `fit` before requesting evaluation metrics.")
        if y_true.shape[0] != self.labels_.shape[0]:
            raise ValueError("Ground truth labels must align with fitted predictions.")

        mask = self.labels_ != -1
        if not np.any(mask):
            raise ValueError("All samples were marked as noise; cannot compute metrics.")

        filtered_true = y_true[mask]
        filtered_pred = self.labels_[mask]
        return {
            "acc": float(metrics.acc(filtered_true, filtered_pred)),
            "nmi": float(metrics.nmi(filtered_true, filtered_pred)),
            "ari": float(metrics.ari(filtered_true, filtered_pred)),
        }

    def get_estimator(self) -> DBSCAN:
        if self._estimator is None:
            raise RuntimeError("Call `fit` before requesting the underlying estimator.")
        return self._estimator

    def evaluate_full(self, X: Optional[np.ndarray], y_true: np.ndarray, save_csv: Optional[str] = None, metric: str = "euclidean") -> Dict[str, float]:
        """Compute a richer set of clustering evaluation metrics.

        Parameters
        ----------
        X: Optional[np.ndarray]
            Feature matrix used to compute silhouette score. If None, silhouette
            will be NaN.
        y_true: np.ndarray
            Ground-truth labels aligned with the training data.
        save_csv: Optional[str]
            If provided, save the metrics to this CSV path (two columns: metric,value).
        metric: str
            The distance metric to pass to silhouette_score.

        Returns
        -------
        Dict[str, float]
            Dictionary of computed metrics.
        """
        if self.labels_ is None:
            raise RuntimeError("Call `fit` before requesting evaluation metrics.")
        if y_true.shape[0] != self.labels_.shape[0]:
            raise ValueError("Ground truth labels must align with fitted predictions.")

        mask = self.labels_ != -1
        if not np.any(mask):
            raise ValueError("All samples were marked as noise; cannot compute metrics.")

        yt = y_true[mask]
        yp = self.labels_[mask]

        results: Dict[str, float] = {}
        # Existing and common metrics
        results["acc"] = float(metrics.acc(yt, yp))
        results["nmi"] = float(normalized_mutual_info_score(yt, yp))
        results["ari"] = float(adjusted_rand_score(yt, yp))
        results["homogeneity"] = float(homogeneity_score(yt, yp))
        results["completeness"] = float(completeness_score(yt, yp))
        results["vmeasure"] = float(v_measure_score(yt, yp))

        # Silhouette: requires X and at least 2 clusters
        if X is None:
            results["silhouette"] = float("nan")
        else:
            Xf = X[mask]
            unique_labels = np.unique(yp)
            if unique_labels.size < 2:
                results["silhouette"] = float("nan")
            else:
                results["silhouette"] = float(silhouette_score(Xf, yp, metric=metric))

        if save_csv is not None:
            os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
            with open(save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in results.items():
                    writer.writerow([k, v])

        return results

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project new samples using the scaling learnt during training."""
        if not self.scale:
            return x
        if self._scaler is None:
            raise RuntimeError("Call `fit` before requesting scaled features.")
        return self._scaler.transform(x)
