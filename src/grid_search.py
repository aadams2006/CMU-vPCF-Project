import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Iterable, List, Tuple

from src.DEC import DEC
from src.IDEC import IDEC
from src.metrics import nmi, ari, acc
from sklearn.metrics import make_scorer

# Helper function to load data (remains unchanged)
def load_vpcf_dataset(
    paths: Iterable[str],
    limit_per_label: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    y_str = np.asarray(labels, dtype=str)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    
    return x, y, y_str

# --- SciKit-Learn Wrapper Classes ---

class DECWrapper(BaseEstimator, ClusterMixin):
    """Wrapper for DEC model to be compatible with GridSearchCV."""
    def __init__(self, dims, n_clusters, alpha=1.0, learning_rate=0.001,
                 pretrain_epochs=10, maxiter=2e4, update_interval=140, batch_size=256):
        self.dims = dims
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.model_ = None

    def fit(self, X, y=None):
        self.model_ = DEC(dims=self.dims, n_clusters=self.n_clusters, alpha=self.alpha)
        self.model_.pretrain(X, optimizer='adam', epochs=self.pretrain_epochs, batch_size=self.batch_size)
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='kld')
        self.labels_ = self.model_.fit(X, y=y, batch_size=self.batch_size, maxiter=self.maxiter, update_interval=self.update_interval)
        return self

    def predict(self, X):
        # The fit method already returns predictions, but for compatibility,
        # we can use the trained model to predict again if needed.
        # In this workflow, we rely on labels_ from fit.
        return self.model_.predict_clusters(X)

class IDECWrapper(BaseEstimator, ClusterMixin):
    """Wrapper for IDEC model to be compatible with GridSearchCV."""
    def __init__(self, dims, n_clusters, alpha=1.0, learning_rate=0.001,
                 pretrain_epochs=10, maxiter=2e4, update_interval=140, batch_size=256):
        self.dims = dims
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.model_ = None

    def fit(self, X, y=None):
        self.model_ = IDEC(dims=self.dims, n_clusters=self.n_clusters, alpha=self.alpha)
        self.model_.pretrain(X, optimizer='adam', epochs=self.pretrain_epochs, batch_size=self.batch_size)
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
        self.labels_ = self.model_.fit(X, y=y, batch_size=self.batch_size, maxiter=self.maxiter, update_interval=self.update_interval)
        return self

    def predict(self, X):
        return self.model_.predict(X)

def run_grid_search():
    """
    Performs a grid search for IDEC and DEC models using GridSearchCV.
    """
    db_path = 'data/saved_vpcfs/HfO_vPCFs_from_images_cif_maxIndex1.json'
    x, y, _ = load_vpcf_dataset([db_path])
    n_clusters = len(np.unique(y))

    # Define the parameter grid
    param_grid = {
        'dims': [[x.shape[-1], 500, 500, 2000, n_clusters], [x.shape[-1], 256, 512, 1024, n_clusters]],
        'alpha': [1.0, 0.5],
        'learning_rate': [0.001, 0.0001]
    }

    # Define scoring metrics
    scoring = {
        'nmi': make_scorer(nmi),
        'ari': make_scorer(ari),
        'acc': make_scorer(acc)
    }

    # To replicate the original behavior of training and testing on the full dataset,
    # we use a PredefinedSplit that uses all data for a single "test" fold.
    # GridSearchCV requires a CV iterator, this is how we provide one without data splitting.
    ps = PredefinedSplit([-1] * len(x))

    results_list = []

    # --- Grid search for DEC ---
    print("Starting Grid Search for DEC...")
    dec_wrapper = DECWrapper(dims=None, n_clusters=n_clusters) # Dims is a search param
    gs_dec = GridSearchCV(
        estimator=dec_wrapper,
        param_grid=param_grid,
        scoring=scoring,
        refit='nmi',  # Refit the best model based on NMI score
        cv=ps,        # Use the predefined split
        n_jobs=-1,    # Use all available CPU cores
        verbose=3
    )
    gs_dec.fit(x, y)
    
    # Process and store DEC results
    dec_results_df = pd.DataFrame(gs_dec.cv_results_)
    dec_results_df['model'] = 'DEC'
    results_list.append(dec_results_df)

    print("\nBest DEC Params:")
    print(gs_dec.best_params_)
    print(f"Best DEC NMI Score: {gs_dec.best_score_:.4f}")


    # --- Grid search for IDEC ---
    print("\nStarting Grid Search for IDEC...")
    idec_wrapper = IDECWrapper(dims=None, n_clusters=n_clusters) # Dims is a search param
    gs_idec = GridSearchCV(
        estimator=idec_wrapper,
        param_grid=param_grid,
        scoring=scoring,
        refit='nmi',
        cv=ps,
        n_jobs=-1,
        verbose=3
    )
    gs_idec.fit(x, y)

    # Process and store IDEC results
    idec_results_df = pd.DataFrame(gs_idec.cv_results_)
    idec_results_df['model'] = 'IDEC'
    results_list.append(idec_results_df)

    print("\nBest IDEC Params:")
    print(gs_idec.best_params_)
    print(f"Best IDEC NMI Score: {gs_idec.best_score_:.4f}")

    # --- Save combined results ---
    # Combine results and select relevant columns
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Standardize parameter column name
    all_results.rename(columns={'params': 'params_str'}, inplace=True)
    all_results['params'] = all_results['params_str'].astype(str)

    # Select and rename columns to match old format
    final_df = all_results[['model', 'params', 'mean_test_nmi', 'mean_test_ari', 'mean_test_acc']].rename(columns={
        'mean_test_nmi': 'nmi',
        'mean_test_ari': 'ari',
        'mean_test_acc': 'acc'
    })

    final_df.to_csv('grid_search_results.csv', index=False)
    print("\nGrid search complete. Results saved to grid_search_results.csv")
    print("\nTop 5 results:")
    print(final_df.sort_values('nmi', ascending=False).head())


if __name__ == "__main__":
    run_grid_search()
