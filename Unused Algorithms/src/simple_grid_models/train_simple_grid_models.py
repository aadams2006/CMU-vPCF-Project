
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

from .DBSCANModel import DBSCANClustering
from .DEC import DEC
from .IDEC import IDEC

def main():
    # Load the dataset
    data_path = r"C:\Users\alexg\Downloads\CMU vPCF Project\simple_grids\simple_grid_features.csv"
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
    print("--- Training DBSCAN ---")
    dbscan = DBSCANClustering(eps=0.5, min_samples=5, scale=True)
    dbscan_labels = dbscan.fit(x)
    print("DBSCAN Labels:", dbscan_labels)
    if len(np.unique(dbscan_labels)) > 1:
        metrics = dbscan.evaluate(y_true)
        print("\n--- DBSCAN Metrics ---")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Normalized Mutual Information: {metrics['nmi']:.4f}")
        print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
        print("-" * 30)
    else:
        print("DBSCAN produced a single cluster or only noise.")

    # --- Train and Evaluate DEC ---
    print("\n--- Training DEC ---")
    dims = [x.shape[1], 500, 500, 2000, len(np.unique(y_true))]
    dec = DEC(dims=dims, n_clusters=len(np.unique(y_true)))
    dec.pretrain(x, epochs=50, batch_size=256)
    dec.compile(optimizer='sgd')
    dec_labels = dec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("DEC Labels:", dec_labels)
    
    # --- DEC Metrics ---
    print("\n--- DEC Metrics (from log file) ---")
    dec_log_file = "results/dec/dec_log.csv"
    if pd.io.common.file_exists(dec_log_file):
        dec_metrics = pd.read_csv(dec_log_file)
        print(dec_metrics.tail(1))
    else:
        print(f"Log file not found: {dec_log_file}")
    print("-" * 30)


    # --- Train and Evaluate IDEC ---
    print("\n--- Training IDEC ---")
    idec = IDEC(dims=dims, n_clusters=len(np.unique(y_true)))
    if not idec.pretrained:
        idec.pretrain(x, epochs=50, batch_size=256)
    idec.compile(optimizer='sgd')
    idec_labels = idec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("IDEC Labels:", idec_labels)

    # --- IDEC Metrics ---
    print("\n--- IDEC Metrics (from log file) ---")
    idec_log_file = "results/idec/idec_log.csv"
    if pd.io.common.file_exists(idec_log_file):
        idec_metrics = pd.read_csv(idec_log_file)
        print(idec_metrics.tail(1))
    else:
        print(f"Log file not found: {idec_log_file}")
    print("-" * 30)

if __name__ == "__main__":
    main()
