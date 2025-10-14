"""DBSCAN clustering utilities for the project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from . import metrics


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

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project new samples using the scaling learnt during training."""
        if not self.scale:
            return x
        if self._scaler is None:
            raise RuntimeError("Call `fit` before requesting scaled features.")
        return self._scaler.transform(x)
