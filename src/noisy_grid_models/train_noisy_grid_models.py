import os
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .DBSCANModel import DBSCANClustering
from .DEC import DEC
from .IDEC import IDEC


def _format_params(params: Dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def optimize_dbscan(
    x: np.ndarray,
    y_true: np.ndarray,
    param_grid: Dict[str, Iterable[object]] | None = None,
    ranking_metrics: Tuple[str, ...] = ("nmi", "silhouette", "ari"),
    results_path: str | None = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Run an exhaustive grid-search for DBSCAN hyper-parameters.

    Parameters
    ----------
    x:
        Feature matrix.
    y_true:
        Ground-truth integer labels.
    param_grid:
        Mapping from parameter name to iterable of candidate values.
    ranking_metrics:
        Ordered tuple describing how to rank successful experiments.
    results_path:
        If provided, the per-trial results will be saved as a CSV file.
    """

    if param_grid is None:
        param_grid = {
            "eps": np.linspace(0.2, 1.0, num=9),
            "min_samples": [3, 5, 10, 20],
            "scale": [True, False],
            "metric": ["euclidean", "manhattan"],
        }

    trials: List[Dict[str, object]] = []
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[key] for key in keys]))

    print(f"Running DBSCAN hyper-parameter search across {len(combos)} configurations...")

    for values in combos:
        params = dict(zip(keys, values))
        try:
            model = DBSCANClustering(**params)
        except TypeError as exc:  # safeguard against invalid params
            print(f"  Skipping invalid parameters {_format_params(params)}: {exc}")
            continue

        try:
            labels = model.fit(x)
        except Exception as exc:  # noqa: BLE001 - surfacing unexpected numerical issues
            print(f"  DBSCAN failed for {_format_params(params)}: {exc}")
            continue

        cluster_labels = np.unique(labels)
        num_clusters = np.sum(cluster_labels != -1)
        if num_clusters < 1:
            print(f"  Configuration {_format_params(params)} produced only noise; skipping.")
            continue

        try:
            metrics = model.evaluate_full(x, y_true)
        except ValueError as exc:
            print(f"  Metrics failed for {_format_params(params)}: {exc}")
            continue

        record: Dict[str, object] = {**params}
        record.update(metrics)
        record["num_clusters"] = int(num_clusters)
        trials.append(record)

    if not trials:
        raise RuntimeError("DBSCAN hyper-parameter search did not yield any valid configuration.")

    df = pd.DataFrame(trials)
    sort_columns = [metric for metric in ranking_metrics if metric in df.columns]
    df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)

    if results_path is not None:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        df.to_csv(results_path, index=False)

    best_params: Dict[str, object] = {}
    for key in keys:
        if key not in df.columns:
            continue
        value = df.iloc[0][key]
        if isinstance(value, np.generic):
            value = value.item()
        best_params[key] = value
    print("Best DBSCAN parameters:")
    print("  " + _format_params(best_params))
    print("Associated metrics:")
    for metric in sort_columns:
        print(f"  {metric}: {df.iloc[0][metric]:.4f}")

    return best_params, df

def main():
    # Load the dataset
    data_path = "noisy_simple_grids_hard/gaussian/noisy_grid_features_gaussian.csv"
    df = pd.read_csv(data_path)

    # Prepare the data
    # --- Specify Model Features and Targets ---
    # Features: 4 histogram values (h1-h4) and 16 co-occurrence values (co01-co16)
    # Targets: 'regime' column, label encoded
    feature_cols = [f'h{i+1}' for i in range(4)] + [f'co{i+1:02d}' for i in range(16)]
    x = df[feature_cols].values.astype('float32')
    y_true_labels = df['regime'].values
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(y_true_labels)

    print("--- Model Features and Targets ---")
    print(f"Features shape: {x.shape}")
    print(f"Targets shape: {y_true.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target labels (encoded): {np.unique(y_true)}")
    print("-" * 30)

    # --- Train and Evaluate DBSCAN ---
    print("--- Optimizing and Training DBSCAN ---")
    results_root = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    dbscan_results_dir = os.path.join(results_root, "dbscan")
    os.makedirs(dbscan_results_dir, exist_ok=True)

    search_csv = os.path.join(dbscan_results_dir, "noisy_hyperparameter_search.csv")
    best_params, _ = optimize_dbscan(x, y_true, results_path=search_csv)

    dbscan = DBSCANClustering(**best_params)
    dbscan_labels = dbscan.fit(x)
    print("DBSCAN Labels:", dbscan_labels)
    unique_labels = np.unique(dbscan_labels)
    if np.sum(unique_labels != -1) > 0:
        metrics = dbscan.evaluate_full(
            x,
            y_true,
            save_csv=os.path.join(dbscan_results_dir, "noisy_metrics.csv"),
        )
        print("\n--- Optimized DBSCAN Metrics ---")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Normalized Mutual Information: {metrics['nmi']:.4f}")
        print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
        print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"Metrics saved to {os.path.join(dbscan_results_dir, 'noisy_metrics.csv')}")
        print(f"Hyperparameter search log saved to {search_csv}")
        print("-" * 30)
    else:
        print("DBSCAN produced a single cluster or only noise even after optimization.")

    # --- Train and Evaluate DEC ---
    print("\n--- Training DEC ---")
    dims = [x.shape[1], 500, 500, 2000, len(np.unique(y_true))]
    dec = DEC(dims=dims, n_clusters=len(np.unique(y_true)), save_dir='results/dec/noisy')
    dec.pretrain(x, epochs=50, batch_size=256)
    dec.compile(optimizer='sgd')
    dec_labels = dec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("DEC Labels:", dec_labels)
    
    # --- DEC Metrics ---
    print("\n--- DEC Metrics (from log file) ---")
    dec_log_file = "results/dec/noisy/dec_log.csv"
    if os.path.exists(dec_log_file):
        dec_metrics = pd.read_csv(dec_log_file)
        print(dec_metrics.tail(1))
    else:
        print(f"Log file not found: {dec_log_file}")
    print("-" * 30)


    # --- Train and Evaluate IDEC ---
    print("\n--- Training IDEC ---")
    idec = IDEC(dims=dims, n_clusters=len(np.unique(y_true)), save_dir='results/idec/noisy')
    if not idec.pretrained:
        idec.pretrain(x, epochs=50, batch_size=256)
    idec.compile(optimizer='sgd')
    idec_labels = idec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("IDEC Labels:", idec_labels)

    # --- IDEC Metrics ---
    print("\n--- IDEC Metrics (from log file) ---")
    idec_log_file = "results/idec/noisy/idec_log.csv"
    if os.path.exists(idec_log_file):
        idec_metrics = pd.read_csv(idec_log_file)
        print(idec_metrics.tail(1))
    else:
        print(f"Log file not found: {idec_log_file}")
    print("-" * 30)

if __name__ == "__main__":
    main()
