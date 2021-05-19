import tensorflow as tf
import numpy as np

import time
import warnings

from collections import deque

import dgps_pepmcm.layers as layer_type
import dgps_pepmcm.config as config
from dgps_pepmcm.kernel.gauss import compute_kernel

from utils import memory_used, calculate_ETA_str


class GPNetwork:
    """
    Class representing a deep GP network
    """

    def __init__(self, training_data, training_targets, no_layers, no_hiddens, no_inducing_points, noisy_layers=True,
                 var_noise=1e-5, lik_noise=1e-2, no_samples=20, inducing_points=None, shared_prior=False, alpha=1.0,
                 seed=0, sacred_exp=None):
        """
        Creates a new Deep GP network
            :param ndarray training_data: Training points (X)
            :param ndarray training_targets: Training targets (Y)
            :param integer no_layers: Number of nidden layers
            :param integer no_hiddens: Number of hidden units
            :param integer no_hiddens: Number of inducing points
            :param Boolean noisy_layers: Indicates if the net needs to have a noisy layer between GP layers
            :param float var_noise: If noisy_layers == True we can specity the noise level
            :param float lik_noise: Likelihood noise
            :param integer no_samples: Number of samples to propagate in training
            :param ndarray inducing_points
            :param Boolean shared_prior: Indicates if the prior should be shared for the nodes in a layer
            :param float alpha: Alpha parameter alpha-divergences
            :param integer seed: Random seed
            :param sacred_exp: Variable with sacred experiment information,
                see: http://sacred.readthedocs.io/en/latest/collected_information.html
        """

        # Reset graph
        tf.compat.v1.reset_default_graph()

        # Initializes random seed
        self.seed = seed
        np.random.seed = seed
        tf.compat.v1.set_random_seed(seed)

        # Read training data
        self.training_data = training_data
        self.training_targets = training_targets
        self.inducing_points = inducing_points

        if self.training_data.ndim == 1:
            self.training_data = self.training_data[:, None]

        if self.training_targets.ndim == 1:
            self.training_targets = self.training_targets[:, None]

        gb_size = self.training_data.nbytes / 1024**3
        if (gb_size > 2):
            x1 = tf.constant(self.training_data[0:int(training_data.shape[0]/2),:], config.float_type_tf)
            x2 = tf.constant(self.training_data[0:int(training_data.shape[0]/2),:], config.float_type_tf)
            self.x_running = tf.concat([x1, x2], 0)
        else:
            # self.x_running = tf.cast(self.training_data, config.float_type_tf)
            self.x_running = self.training_data

        # Create placeholders for the data
        self.training_data_tf = tf.compat.v1.placeholder(config.float_type_tf, name="x_training",
                                               shape=[None, self.training_data.shape[1]])
        self.training_targets_tf = tf.compat.v1.placeholder(config.float_type_tf, name="y_training",
                                                  shape=[None, self.training_targets.shape[1]])
        self.set_for_training = tf.compat.v1.placeholder(config.float_type_tf, name="set_for_training", shape=[])

        # Model parameters
        self.no_layers = no_layers
        self.no_hiddens = no_hiddens
        self.no_inducing_points = no_inducing_points
        self.noisy_layers = noisy_layers
        self.var_noise = var_noise
        self.lik_noise = lik_noise
        self.shared_prior = shared_prior
        self.no_points = training_data.shape[0]
        self.problem_dim = training_data.shape[1]
        self.no_samples = {
            'training': no_samples,    # num samples for training
            'prediction': 100         # num samples for prediction
        }
        self.alpha = alpha
        self.sacred_exp = sacred_exp  # This is a useful variable to store sacred experiments data.

        # Classification of regression
        # If targets are integer -> classification problem
        self.is_classification = np.issubdtype(self.training_targets.dtype, np.integer)
        if self.is_classification:
            self.no_classes = np.max(self.training_targets) + 1

        # Build network structure
        self.layers = self._build_net()

        # Build part of the graph that deals with prediction
        self.outputLikelihoodError = self.layers[-1].getLogLikelihoodError()
        self.predict_function = self.layers[-1].getPredictedValues()
        self.sampling_predictive = self.layers[-1].sampleFromPredictive()
        self.predictive_dist = self.layers[-1].getPredictiveDistribution()
        self.sampling_posterior = self.layers[-1].sampleFromPosterior()

        # Create tensorflow session
        self.sess = tf.compat.v1.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def _build_net(self):
        """
        Method that builds the GP network
        """

        # First layer is input layer
        layers = [layer_type.InputLayer(self.training_data_tf, self.problem_dim, self.no_samples, self.set_for_training)]

        # Add GP layers
        for layer_i in range(self.no_layers):
            input_dim_layer = layers[-1].no_nodes
            islast = layer_i == self.no_layers - 1
            if islast:
                if self.is_classification and self.no_classes > 2:
                    output_dim_layer = self.no_classes
                else:
                    output_dim_layer = 1
            else:
                output_dim_layer = self.no_hiddens

            init_dict = self._initialization_dict(layer_i, input_dim_layer, output_dim_layer, islast)

            output_previous = layers[-1].getOutput()
            layers += [layer_type.GPLayer(self.no_inducing_points, self.no_points, output_dim_layer, input_dim_layer,
                                          self.no_samples, self.shared_prior, self.set_for_training, self.alpha, self.seed, init_dict,
                                          output_previous[0], output_previous[1], layer_i)]

            del init_dict

            # Add noisy layer if specified and not in the last GP layer
            if self.noisy_layers and not islast:
                output_previous = layers[-1].getOutput()
                layers += [layer_type.NoisyLayer(self.var_noise, output_dim_layer, output_previous[0], output_previous[1], layer_i)]

        output_previous = layers[-1].getOutput()

        # Output layer depending on the type of problem
        if self.is_classification:
            # If is a multiclass problem
            if self.no_classes > 2:
                layers += [layer_type.OutputLayerMulticlass(self.training_targets_tf, self.no_samples, self.alpha, self.seed, self.no_classes,
                                                        self.lik_noise, output_previous[0], output_previous[1])]
            else:
                layers += [layer_type.OutputLayerClassification(self.training_targets_tf, self.no_samples, self.alpha,  self.seed,
                                                        self.lik_noise, output_previous[0], output_previous[1])]
        else:
            layers += [layer_type.OutputLayerRegression(self.training_targets_tf, self.no_samples, self.alpha, self.seed,
                                                        self.no_points, self.lik_noise, output_previous[0], output_previous[1])]

        return layers

    def _initialization_dict(self, layer_i, input_dim_layer, output_dim_layer, islast=False):
        """
        Define the initial values for the model parameters of a given layer:
            W: Weight matrix
            Z: Inducing points
            lls: Lengthscales RBF kernel
            lsf: Variance RBF kernel
            LParamPost: Covariance of the approximation q
            mParamPost: Mean of the approximation q
        """

        if self.shared_prior:
            output_dim_layer_hyper = 1
        else:
            output_dim_layer_hyper = output_dim_layer

        # set mean function weights of the layer, W should have the same dimension [1]
        # as the number of nodes in the layer
        W = self._linear_mean_function_vi(input_dim_layer, output_dim_layer)
        if islast:
            W = tf.zeros_like(W, dtype=config.float_type_tf)

        if self.inducing_points is not None:
            # Use given initializations
            z = self._adjust_z_dim(input_dim_layer)
        # If we are not in the first layer, we initialize in the -1, 1 range the inducing points
        elif layer_i == 0:
            # First Layer
            z = self.training_data[np.random.choice(self.no_points, size=self.no_inducing_points, replace=False), :]
        else:
            # Not first layer and we don't have any initialization to use
            z = tf.tile(tf.reshape(tf.lin_space(tf.constant(-1.0, dtype=config.float_type_tf), tf.constant(1.0, dtype=config.float_type_tf), self.no_inducing_points),
                                   shape=[self.no_inducing_points, 1]),
                        [1, input_dim_layer])

        z = z * tf.ones([output_dim_layer_hyper, self.no_inducing_points, input_dim_layer], dtype=config.float_type_tf)

        lls = tf.zeros([output_dim_layer_hyper, input_dim_layer], dtype=config.float_type_tf)
        lsf = tf.zeros([output_dim_layer_hyper], dtype=config.float_type_tf)
        
        training_data = tf.constant(self.training_data[np.random.choice(self.no_points, size=self.no_inducing_points, replace=False), :], dtype=config.float_type_tf)
        M = tf.reduce_sum(training_data**2, 1)[:, None] * tf.ones([1, self.no_inducing_points], dtype=config.float_type_tf)
        dist = M - 2 * tf.matmul(training_data, training_data, transpose_b=True) + tf.transpose(M)
        median = tf.contrib.distributions.percentile(dist, q=50)
        lls = tf.math.log(median) * tf.ones(tf.shape(lls), dtype=config.float_type_tf)

        L = tf.random.normal(shape=[output_dim_layer, self.no_inducing_points, self.no_inducing_points], dtype=config.float_type_tf, seed=self.seed)
        if not islast:
            L *= config.jitter
        m = tf.random.normal(shape=[output_dim_layer, self.no_inducing_points, 1], dtype=config.float_type_tf, seed=self.seed)  # seed=8

        return {'W': W, 'inducing_points': z, 'Lengthscales': lls, 'lsf': lsf, 'LParamPost': L, 'mParamPost': m}

    def _linear_mean_function_vi(self, input_dim_layer, output_dim_layer):
        """
        Set the W for the mean function m(X) = XW
        The last GP layer will have m(X) = 0
        This method is based on Doubly Stochastic Variational Inference for Deep Gaussian Processes https://arxiv.org/abs/1705.08933

            :param input_dim_layer: Input dimension of the GP layer
            :param output_dim_layer: Output dimension of the GP layer
        """
        if input_dim_layer == output_dim_layer:
            W = tf.eye(output_dim_layer, dtype=config.float_type_tf)
        elif output_dim_layer > input_dim_layer:
            zeros = tf.zeros((input_dim_layer, output_dim_layer - input_dim_layer), dtype=config.float_type_tf)
            W = tf.concat([tf.eye(input_dim_layer, dtype=config.float_type_tf), zeros], 1)
            # self.x_running = self.x_running.dot(W)
            self.x_running = tf.matmul(self.x_running, W)
        elif output_dim_layer < input_dim_layer:
            _, _, V = tf.linalg.svd(self.x_running)
            W = tf.transpose(V[:output_dim_layer, :])
            self.x_running = tf.matmul(self.x_running, W)
        return W

    def _adjust_z_dim(self, input_dim_layer):
        """
        Adjust the inducing points to the correct dimensions

            :param input_dim_layer: Input dimension of the layer
        """
        assert self.problem_dim == self.inducing_points.shape[1], "Z should have the same dimension as the problem"
        self.inducing_points = tf.cast(self.inducing_points, config.float_type_tf)
        if self.problem_dim == input_dim_layer:
            # Case 1: Same dimensions, nothing to do
            Z = self.inducing_points
        elif self.problem_dim > input_dim_layer:
            # Case 2: Reduce dimensions
            _, _, V = np.linalg.svd(self.training_data, full_matrices=False)  # V -> (D,D) Matrix
            W = V[:input_dim_layer, :].T  # W has only the input_dim_layer principal components of X
            Z = tf.matmul(self.inducing_points, tf.constant(W, dtype=config.float_type_tf))
        elif self.problem_dim < input_dim_layer:
            # Case 3: Increase dimensions. Just tile the first PCA component.
            _, _, V = np.linalg.svd(self.training_data, full_matrices=False)  # V -> (D,D) Matrix
            proj = tf.expand_dims(tf.transpose(tf.constant(V[0, :], dtype=config.float_type_tf)), 1)
            first_pca = tf.squeeze(tf.matmul(self.inducing_points, proj))  # First Principal component
            Z = tf.tile(first_pca[:, None], (1, input_dim_layer))

        return Z

    def getNetworkEnergy(self):
        """
        Builds the graph corresponding to the energy to minimize
        """
        energy = 0.0
        for layer in self.layers:
            energy += layer.getLayerContributionToEnergy()
        return energy

    
    def predict(self, test_data):
        """
        Predict the output for new data

            :param test_data: Data points to predict
            :param y_train_std: Standard deviation of the original data
            :param y_train_mean: Mean of the original data
        """
        assert test_data.shape[1] == self.problem_dim

        test_data = test_data.astype(config.float_type_np)

        predicted_values = self.sess.run(self.predict_function, feed_dict={
                self.training_data_tf: test_data,
                self.set_for_training: 0.0
            })

        return predicted_values

    def sampleFromPredictive(self, points, y_train_std, y_train_mean):
        """
        Sample from the posterior distribution

            :param points: Space from which to sample
            :param y_train_std: Standard deviation of the original data
            :param y_train_mean: Mean of the original data
        """
        assert points.shape[1] == self.problem_dim

        points = points.astype(config.float_type_np)

        output_layer = self.layers[-1]

        samples = self.sess.run(self.sampling_predictive, feed_dict={
                self.training_data_tf: points,
                output_layer.y_train_std: y_train_std,
                output_layer.y_train_mean: y_train_mean,
                self.set_for_training: 0.0
            })

        return samples

    def getPredictiveDistribution(self, x_value, y_range):
        """
        Compute the predictive distribution for given X's in a range

            :param x_value
            :param y_range
        """
        assert x_value.shape[1] == self.problem_dim

        x_value = x_value.astype(config.float_type_np)

        pdf = self.sess.run(self.predictive_dist, feed_dict={
                self.training_data_tf: x_value,
                self.training_targets_tf: y_range,
                self.set_for_training: 0.0
            })

        return pdf

    def getLogLikelihoodError(self, X_test, y_test, y_train_std=None, y_train_mean=None):
        """
        Compute the test log likelihood and test error (rmse for regression)
        We expect unnormalized y_test
        Check Normalized Inputs

            :param X_test
            :param y_test
            :param y_train_std: Standard deviation of the original data
            :param y_train_mean: Mean of the original data
        """
        if not np.allclose(np.mean(X_test), 0, atol=0.1) or not np.allclose(np.std(X_test), 1.0, atol=0.1):
            warnings.warn(f"X_test should be normalized current mean = {np.mean(X_test)} and std = {np.std(X_test)}")

        output_layer = self.layers[-1]

        n_batches = max(int(X_test.shape[0]/100), 1)
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(np.array_split(X_test, n_batches), np.array_split(y_test, n_batches)):
            if self.is_classification: # Classification
                # import pdb; pdb.set_trace()
                l, sq = self.sess.run(self.outputLikelihoodError,
                                feed_dict={
                                self.training_data_tf: X_batch,
                                output_layer.test_targets_tf: Y_batch.astype(config.int_type_np),
                                self.set_for_training: 0.0
                                })
            else:   # Regression
                l, sq = self.sess.run(self.outputLikelihoodError,
                                    feed_dict={
                                        self.training_data_tf: X_batch,
                                        output_layer.test_targets_tf: Y_batch,
                                        output_layer.y_train_std: y_train_std,
                                        output_layer.y_train_mean: y_train_mean,
                                        self.set_for_training: 0.0
                                    })

            lik.append(l)
            sq_diff.append(sq)

        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
        
        if self.is_classification: # Classification
            return np.average(lik), np.average(sq_diff) 
        else:
            return np.average(lik), np.average(sq_diff) ** 0.5


    def train(self, optimizer, no_epochs=500, minibatch_size=100, X_test=None, y_test=None, y_train_std=None, y_train_mean=None,
              show_training_info=True, log_every=10, continue_training=False):
        """
        Start the optimization of the objective function

            :param optimizer: A tensorflow optimizer, e.g. AdamOptimizer
            :param no_epochs: Number of epochs to train
            :param minibatch_size: Size of the minibatches
            :param X_test: If != None it computes test performance during the training process
            :param y_test: If != None it computes test performance during the training process
            :param y_train_std: If != None it computes test performance during the training process
            :param y_train_mean: If != None it computes test performance during the training process
            :param show_training_info: Verbose mode
            :param log_every: Log every x iterations
        """

        assert len(self.layers) > 1

        n_data_points = self.training_data.shape[0]
        self.minibatch_size = minibatch_size

        if not continue_training:
            if show_training_info:
                print('Compiling computation graph')
            self.energy = -self.getNetworkEnergy()

            self.train_step = optimizer.minimize(self.energy)

            if show_training_info:
                print('Initializing network')
            self.sess.run(tf.compat.v1.global_variables_initializer())

            self.sess.graph.finalize()

            if show_training_info:
                print('Training')

        n_batches = int(np.ceil(1.0 * n_data_points / self.minibatch_size))

        start = time.time()
        start_epoch = time.time()
        
        # Object that keeps maxlen epoch times, for ETA prediction.
        last_epoch_times = deque(maxlen=20)

        # Main optimization loop
        gradient_steps = 0
        for _ in range(no_epochs):
            suffle = np.random.choice(n_data_points, n_data_points, replace=False)
            self.training_data = self.training_data[suffle, :]
            self.training_targets = self.training_targets[suffle, :]
            
            avg_energy = 0.0

            for i in range(n_batches):
                minibatch_data = self.training_data[i * self.minibatch_size:min((i + 1) * self.minibatch_size, n_data_points), :]
                minibatch_targets = self.training_targets[i * self.minibatch_size:min((i + 1) * self.minibatch_size, n_data_points), :]
                dict = {self.training_data_tf: minibatch_data, self.training_targets_tf: minibatch_targets, self.set_for_training: 1.0}

                current_energy = self.sess.run([self.energy, self.train_step], dict)[0]

                avg_energy += current_energy

                gradient_steps += 1

                if gradient_steps % log_every == 0:

                    energy_to_log = -avg_energy * n_batches / (i+1)
                    if X_test is not None and y_test is not None:
                        test_nll, test_error = self.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)
                    if show_training_info:
                        elapsed_time_epoch = time.time() - start_epoch
                        last_epoch_times.append(elapsed_time_epoch)
                        eta = calculate_ETA_str(last_epoch_times, gradient_steps, no_epochs * n_batches)

                        if X_test is not None and y_test is not None:

                            if self.is_classification: # Classification
                                print(
                                        "Iteration: {: <4}| Energy: {: <11.6f} | Time: {: >8.4f}s | Memory: {: >2.2f} GB | ETA: {} | TestNLL: {: >5.6}, TestError: {: >5.6}".format(
                                         gradient_steps, energy_to_log, elapsed_time_epoch, memory_used(), eta, test_nll, test_error
                                     )
                                 )
                            else:
                                print(
                                        "Iteration: {: <4}| Energy: {: <11.6f} | Time: {: >8.4f}s | Memory: {: >2.2f} GB | ETA: {} | TestNLL: {: >5.6}, TestRMSE: {: >5.6}".format(
                                         gradient_steps, energy_to_log, elapsed_time_epoch, memory_used(), eta, test_nll, test_error
                                     )
                                 )
                            
                        else:
                            print(
                                 "Iteration: {: <4}| Energy: {: <11.6f} | Time: {: >8.4f}s | Memory: {: >2.2f} GB | ETA: {}".format(
                                     gradient_steps, energy_to_log, elapsed_time_epoch, memory_used(), eta
                                 )
                             )
                    else:
                        elapsed_time_epoch = time.time() - start_epoch

                    if self.sacred_exp is not None:
                        if X_test is not None and y_test is not None:
                            self.sacred_exp.log_scalar("test.nll", -test_nll)
                            self.sacred_exp.log_scalar("test.rmse", test_error)
                            self.sacred_exp.log_scalar("time", elapsed_time_epoch)
                        self.sacred_exp.log_scalar("train.energy", round(energy_to_log, 4))

                    # start_epoch = time.process_time()
                    start_epoch = time.time()

        # elapsed_time = time.process_time() - start
        elapsed_time = time.time() - start

        if show_training_info:
            print("Total time: {}".format(elapsed_time))

        # Log final energy
        if self.sacred_exp is not None:
            if self.sacred_exp.info.get('last_train_energies') is None:
                self.sacred_exp.info.update({"last_train_energies": [round(avg_energy, 4)]})
            else:
                self.sacred_exp.info.get('last_train_energies').append(round(avg_energy, 4))
