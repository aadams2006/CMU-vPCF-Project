from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import os


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` which represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: soft labels for each sample, shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self, dims, n_clusters=10, alpha=1.0, init='glorot_uniform', save_dir='results/dec'):
        super(DEC, self).__init__()
        self.dims = dims
        self.input_dim = self.dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = self.build_autoencoder(init)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.pretrained = False

    def build_autoencoder(self, init):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        # input
        x = Input(shape=(self.input_dim,), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks - 1):
            h = Dense(self.dims[i + 1], activation='relu', kernel_initializer=init, name='encoder_%d' % i)(h)

        # hidden layer
        h = Dense(self.dims[-1], kernel_initializer=init, name='encoder_%d' % (self.n_stacks - 1))(h)

        y = h
        # internal layers in decoder
        for i in range(self.n_stacks - 1, 0, -1):
            y = Dense(self.dims[i], activation='relu', kernel_initializer=init, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=init, name='decoder_0')(y)

        return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # begin pretraining
        from time import time
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: ', time() - t0)
        weights_path = os.path.join(self.save_dir, 'ae_weights.weights.h5')
        self.autoencoder.save_weights(weights_path)
        print('Pretrained weights are saved to %s' % weights_path)
        self.pretrained = True

    def load_weights(self, weights_path): # load weights of DEC model
        self.model.load_weights(weights_path)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q): # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3, update_interval=140):
        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5

        # Step 1: pretrain if necessary
        if not self.pretrained:
            self.pretrain(x, batch_size=batch_size)

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv
        logfile = open(os.path.join(self.save_dir, 'dec_log.csv'), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=p[index * batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=p[index * batch_size:(index + 1) * batch_size])
                index += 1


            # save intermediate model
            if ite % save_interval == 0:
                # save DEC model checkpoints
                checkpoint_path = os.path.join(self.save_dir, f'DEC_model_{ite}.weights.h5')
                print('saving model to:', checkpoint_path)
                self.model.save_weights(checkpoint_path)

            ite += 1

        # save the trained model
        logfile.close()
        final_path = os.path.join(self.save_dir, 'DEC_model_final.weights.h5')
        print('saving model to:', final_path)
        self.model.save_weights(final_path)

        return y_pred