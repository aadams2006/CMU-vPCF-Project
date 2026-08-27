"""Keras implementation of Improved Deep Embedded Clustering for phase-boundary training."""

from __future__ import annotations

import csv
import os
from time import time

import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

try:
    from . import metrics as cluster_metrics
    from .dec import ClusteringLayer
except ImportError:
    import metrics as cluster_metrics
    from dec import ClusteringLayer


def autoencoder(dims, act: str = "relu", init: str = "glorot_uniform"):
    """Build a symmetric fully connected autoencoder."""
    n_stacks = len(dims) - 1
    inputs = Input(shape=(dims[0],), name="input")
    hidden = inputs

    for layer_idx in range(n_stacks - 1):
        hidden = Dense(
            dims[layer_idx + 1],
            activation=act,
            kernel_initializer=init,
            name=f"encoder_{layer_idx}",
        )(hidden)

    hidden = Dense(dims[-1], kernel_initializer=init, name=f"encoder_{n_stacks - 1}")(hidden)

    decoded = hidden
    for layer_idx in range(n_stacks - 1, 0, -1):
        decoded = Dense(
            dims[layer_idx],
            activation=act,
            kernel_initializer=init,
            name=f"decoder_{layer_idx}",
        )(decoded)

    decoded = Dense(dims[0], kernel_initializer=init, name="decoder_0")(decoded)
    return Model(inputs=inputs, outputs=decoded, name="AE"), Model(
        inputs=inputs,
        outputs=hidden,
        name="encoder",
    )


class IDEC:
    """Improved Deep Embedded Clustering model wrapper."""

    def __init__(
        self,
        dims,
        n_clusters: int = 10,
        alpha: float = 1.0,
        gamma: float = 0.1,
        init: str = "glorot_uniform",
        save_dir: str = "results/idec",
        random_state: int = 42,
    ):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = int(random_state)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(self.encoder.output)
        self.model = Model(
            inputs=self.encoder.input,
            outputs=[clustering_layer, self.autoencoder.output],
        )
        self.pretrained = False

    def pretrain(self, x, optimizer: str = "adam", epochs: int = 200, batch_size: int = 256):
        """Pretrain the autoencoder reconstruction objective."""
        print("...Pretraining IDEC autoencoder...")
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=1)
        print("Pretraining time:", time() - t0)
        weights_path = os.path.join(self.save_dir, "ae_weights.weights.h5")
        self.autoencoder.save_weights(weights_path)
        print(f"Pretrained weights saved to {weights_path}")
        self.pretrained = True

    def load_weights(self, weights_path: str):
        self.model.load_weights(weights_path)

    def extract_features(self, x):
        return self.encoder.predict(x, verbose=0)

    def get_cluster_centers(self):
        return self.model.get_layer(name="clustering").get_weights()[0]

    def predict_clusters(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer: str = "sgd", loss=None, loss_weights=None):
        if loss is None:
            loss = ["kld", "mse"]
        if loss_weights is None:
            loss_weights = [self.gamma, 1.0]
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def fit(
        self,
        x,
        y=None,
        maxiter: int = int(2e4),
        batch_size: int = 256,
        tol: float = 1e-3,
        update_interval: int = 140,
    ):
        print("Update interval", update_interval)
        save_interval = max(1, int(np.ceil(x.shape[0] / batch_size * 5)))

        if not self.pretrained:
            self.pretrain(x, batch_size=batch_size)

        print("Initializing cluster centers with k-means.")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state,
        )
        y_pred = kmeans.fit_predict(self.encoder.predict(x, verbose=0))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        log_path = os.path.join(self.save_dir, "idec_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=["iter", "acc", "nmi", "ari", "L", "Lc", "Lr"])
            logwriter.writeheader()

            loss = [0.0, 0.0, 0.0]
            rng = np.random.default_rng(self.random_state)
            for iteration in range(int(maxiter)):
                if iteration % update_interval == 0:
                    q, _ = self.model.predict(x, verbose=0)
                    p = self.target_distribution(q)
                    y_pred = q.argmax(1)

                    if y is not None:
                        acc = np.round(cluster_metrics.acc(y, y_pred), 5)
                        nmi = np.round(cluster_metrics.nmi(y, y_pred), 5)
                        ari = np.round(cluster_metrics.ari(y, y_pred), 5)
                        rounded_loss = np.round(loss, 5)
                        logwriter.writerow(
                            {
                                "iter": iteration,
                                "acc": acc,
                                "nmi": nmi,
                                "ari": ari,
                                "L": rounded_loss[0],
                                "Lc": rounded_loss[1],
                                "Lr": rounded_loss[2],
                            }
                        )
                        print(
                            f"Iter {iteration}: acc = {acc:.5f}, nmi = {nmi:.5f}, "
                            f"ari = {ari:.5f} ; loss = {rounded_loss}"
                        )

                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if iteration > 0 and delta_label < tol:
                        print(f"delta_label {delta_label} < tol {tol}")
                        print("Reached tolerance threshold. Stopping training.")
                        break

                batch_idx = rng.choice(x.shape[0], batch_size)
                loss = self.model.train_on_batch(x=x[batch_idx], y=[p[batch_idx], x[batch_idx]])

                if iteration % save_interval == 0:
                    checkpoint_path = os.path.join(self.save_dir, f"IDEC_model_{iteration}.weights.h5")
                    print("Saving model checkpoint to:", checkpoint_path)
                    self.model.save_weights(checkpoint_path)

        final_path = os.path.join(self.save_dir, "IDEC_model_final.weights.h5")
        print("Saving final IDEC model to:", final_path)
        self.model.save_weights(final_path)
        final_q, _ = self.model.predict(x, verbose=0)
        return final_q.argmax(1)
