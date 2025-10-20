
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
    feature_cols = [f'h{i+1}' for i in range(4)] + [f'co{i+1:02d}' for i in range(16)]
    x = df[feature_cols].values.astype('float32')
    y_true_labels = df['regime'].values
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(y_true_labels)

    # --- Train and Evaluate DBSCAN ---
    print("--- Training DBSCAN ---")
    dbscan = DBSCANClustering(eps=0.5, min_samples=5, scale=True)
    dbscan_labels = dbscan.fit(x)
    print("DBSCAN Labels:", dbscan_labels)
    if len(np.unique(dbscan_labels)) > 1:
        metrics = dbscan.evaluate(y_true)
        print("DBSCAN Metrics:", metrics)
    else:
        print("DBSCAN produced a single cluster or only noise.")

    # --- Train and Evaluate DEC ---
    print("\n--- Training DEC ---")
    dims = [x.shape[1], 500, 500, 2000, 10]
    dec = DEC(dims=dims, n_clusters=len(np.unique(y_true)))
    dec.pretrain(x, epochs=50, batch_size=256)
    dec.compile(optimizer='sgd')
    dec_labels = dec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("DEC Labels:", dec_labels)

    # --- Train and Evaluate IDEC ---
    print("\n--- Training IDEC ---")
    idec = IDEC(dims=dims, n_clusters=len(np.unique(y_true)))
    if not idec.pretrained:
        idec.pretrain(x, epochs=50, batch_size=256)
    idec.compile(optimizer='sgd')
    idec_labels = idec.fit(x, y=y_true, maxiter=8000, update_interval=200, batch_size=256)
    print("IDEC Labels:", idec_labels)

if __name__ == "__main__":
    main()
