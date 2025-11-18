import json
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.DEC import DEC
from src.IDEC import IDEC
from src.metrics import nmi, ari, acc
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from typing import Iterable, List, Tuple

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
    y_str = np.asarray(labels, dtype=str)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    
    return x, y

def run_grid_search():
    """
    Performs a grid search for IDEC and DEC models to find the best hyperparameters.
    """
    db_path = 'data/saved_vpcfs/HfO_vPCFs_from_images_cif_maxIndex1.json'
    x, y = load_vpcf_dataset([db_path])
    n_clusters = len(np.unique(y))

    # Define the parameter grid
    param_grid = {
        'dims': [[x.shape[-1], 500, 500, 2000, n_clusters], [x.shape[-1], 256, 512, 1024, n_clusters]],
        'alpha': [1.0, 0.5],
        'learning_rate': [0.001, 0.0001]
    }

    results = []

    # Grid search for DEC
    print("Starting Grid Search for DEC...")
    for params in ParameterGrid(param_grid):
        print(f"Testing DEC with params: {params}")
        dec = DEC(dims=params['dims'], n_clusters=n_clusters, alpha=params['alpha'])
        dec.pretrain(x, optimizer='adam', epochs=10, batch_size=256)
        dec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='kld')
        y_pred = dec.fit(x, y=y, batch_size=256, maxiter=2e4, update_interval=140)
        
        nmi_score = nmi(y, y_pred)
        ari_score = ari(y, y_pred)
        acc_score = acc(y, y_pred)
        
        results.append({
            'model': 'DEC',
            'params': str(params),
            'nmi': nmi_score,
            'ari': ari_score,
            'acc': acc_score
        })
        print(f"DEC scores: NMI={nmi_score:.4f}, ARI={ari_score:.4f}, ACC={acc_score:.4f}")

    # Grid search for IDEC
    print("\nStarting Grid Search for IDEC...")
    for params in ParameterGrid(param_grid):
        print(f"Testing IDEC with params: {params}")
        idec = IDEC(dims=params['dims'], n_clusters=n_clusters, alpha=params['alpha'])
        idec.pretrain(x, optimizer='adam', epochs=10, batch_size=256)
        idec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
        y_pred = idec.fit(x, y=y, batch_size=256, maxiter=2e4, update_interval=140)
        
        nmi_score = nmi(y, y_pred)
        ari_score = ari(y, y_pred)
        acc_score = acc(y, y_pred)
        
        results.append({
            'model': 'IDEC',
            'params': str(params),
            'nmi': nmi_score,
            'ari': ari_score,
            'acc': acc_score
        })
        print(f"IDEC scores: NMI={nmi_score:.4f}, ARI={ari_score:.4f}, ACC={acc_score:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('grid_search_results.csv', index=False)
    print("\nGrid search complete. Results saved to grid_search_results.csv")

    # Find best params
    best_dec_params = results_df[results_df['model'] == 'DEC'].loc[results_df[results_df['model'] == 'DEC']['nmi'].idxmax()]
    best_idec_params = results_df[results_df['model'] == 'IDEC'].loc[results_df[results_df['model'] == 'IDEC']['nmi'].idxmax()]

    print("\nBest DEC Params:")
    print(best_dec_params)
    print("\nBest IDEC Params:")
    print(best_idec_params)

if __name__ == "__main__":
    import tensorflow as tf
    run_grid_search()