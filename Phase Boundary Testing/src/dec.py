"""Keras implementation of Deep Embedded Clustering for phase-boundary training."""

from __future__ import annotations

import csv
import os
from time import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, InputSpec, Layer
from tensorflow.keras.models import Model

try:
    from . import metrics as cluster_metrics
except ImportError:
    import metrics as cluster_metrics


class ClusteringLayer(Layer):
    """Student t-distribution clustering layer used by DEC and IDEC."""

    def __init__(self, n_clusters: int, weights=None, alpha: float = 1.0, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("ClusteringLayer expects rank-2 inputs.")
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            self.initial_weights = None
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (
            1.0
            + (
                K.sum(
                    K.square(K.expand_dims(inputs, axis=1) - self.clusters),
                    axis=2,
                )
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters, "alpha": self.alpha}
        base_config = super().get_config()
        return {**base_config, **config}


class DEC:
    """Deep Embedded Clustering model wrapper."""

    def __init__(
        self,
        dims,
        n_clusters: int = 10,
        alpha: float = 1.0,
        init: str = "glorot_uniform",
        save_dir: str = "results/dec",
        random_state: int = 42,
    ):
        self.dims = dims
        self.input_dim = self.dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.random_state = int(random_state)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.autoencoder, self.encoder = self.build_autoencoder(init)
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.pretrained = False

    def build_autoencoder(self, init: str):
        """Build a fully connected symmetric autoencoder."""
        inputs = Input(shape=(self.input_dim,), name="input")
        hidden = inputs

        for layer_idx in range(self.n_stacks - 1):
            hidden = Dense(
                self.dims[layer_idx + 1],
                activation="relu",
                kernel_initializer=init,
                name=f"encoder_{layer_idx}",
            )(hidden)

        hidden = Dense(
            self.dims[-1],
            kernel_initializer=init,
            name=f"encoder_{self.n_stacks - 1}",
        )(hidden)

        decoded = hidden
        for layer_idx in range(self.n_stacks - 1, 0, -1):
            decoded = Dense(
                self.dims[layer_idx],
                activation="relu",
                kernel_initializer=init,
                name=f"decoder_{layer_idx}",
            )(decoded)

        decoded = Dense(self.dims[0], kernel_initializer=init, name="decoder_0")(decoded)
        return Model(inputs=inputs, outputs=decoded, name="AE"), Model(
            inputs=inputs,
            outputs=hidden,
            name="encoder",
        )

    def pretrain(self, x, optimizer: str = "adam", epochs: int = 200, batch_size: int = 256):
        """Pretrain the autoencoder reconstruction objective."""
        print("...Pretraining DEC autoencoder...")
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
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer: str = "sgd", loss: str = "kld"):
        self.model.compile(optimizer=optimizer, loss=loss)

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

        log_path = os.path.join(self.save_dir, "dec_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=["iter", "acc", "nmi", "ari", "loss"])
            logwriter.writeheader()

            loss = 0.0
            index = 0
            for iteration in range(int(maxiter)):
                if iteration % update_interval == 0:
                    q = self.model.predict(x, verbose=0)
                    p = self.target_distribution(q)
                    y_pred = q.argmax(1)

                    if y is not None:
                        acc = np.round(cluster_metrics.acc(y, y_pred), 5)
                        nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
                        ari = np.round(adjusted_rand_score(y, y_pred), 5)
                        loss_value = float(np.round(loss, 5))
                        logwriter.writerow(
                            {"iter": iteration, "acc": acc, "nmi": nmi, "ari": ari, "loss": loss_value}
                        )
                        print(
                            f"Iter {iteration}: acc = {acc:.5f}, nmi = {nmi:.5f}, "
                            f"ari = {ari:.5f} ; loss = {loss_value}"
                        )

                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if iteration > 0 and delta_label < tol:
                        print(f"delta_label {delta_label} < tol {tol}")
                        print("Reached tolerance threshold. Stopping training.")
                        break

                if (index + 1) * batch_size > x.shape[0]:
                    loss = self.model.train_on_batch(x=x[index * batch_size :], y=p[index * batch_size :])
                    index = 0
                else:
                    loss = self.model.train_on_batch(
                        x=x[index * batch_size : (index + 1) * batch_size],
                        y=p[index * batch_size : (index + 1) * batch_size],
                    )
                    index += 1

                if iteration % save_interval == 0:
                    checkpoint_path = os.path.join(self.save_dir, f"DEC_model_{iteration}.weights.h5")
                    print("Saving model checkpoint to:", checkpoint_path)
                    self.model.save_weights(checkpoint_path)

        final_path = os.path.join(self.save_dir, "DEC_model_final.weights.h5")
        print("Saving final DEC model to:", final_path)
        self.model.save_weights(final_path)
        final_q = self.model.predict(x, verbose=0)
        return final_q.argmax(1)
