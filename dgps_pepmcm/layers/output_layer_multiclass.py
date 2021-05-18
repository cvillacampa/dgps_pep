
import numpy as np
import tensorflow as tf

import dgps_pepmcm.config as config
from dgps_pepmcm.layers import BaseLayer


class OutputLayerMulticlass(BaseLayer):
    """
    This file contains the object that represents an input layer to the network
    """

    def __init__(self, training_targets, n_samples, alpha, seed, n_classes, lik_noise, input_means, input_vars):
        assert (
            n_classes > 2
        ), "The number of classes should be greater than 2 to use the layer Multiclass"

        # TODO: n_classes = D  so we don't really need the parameter
        BaseLayer.__init__(self)

        self.training_targets = training_targets
        self.alpha = alpha
        self.seed = seed
        self.input_means = input_means
        self.input_vars = input_vars
        self.lvar_noise = tf.get_variable("lik_noise", initializer=tf.log(tf.constant(lik_noise, dtype=config.float_type_tf)), dtype=config.float_type_tf)
        self.test_targets_tf = tf.placeholder(config.int_type_tf, name="y_test", shape=[None, 1])


        self.n_classes = n_classes
        self.grid_size = 100
        self.norm = tf.distributions.Normal(
            loc=tf.constant(0.0, config.float_type_tf), scale=tf.constant(1.0, config.float_type_tf)
        )

        # Latent variable that represents noise in the labels
        self.latent_prob_wrong_label = tf.constant(self.logit(1e-3), dtype=config.float_type_tf)


    @classmethod
    def logit(self, p):
        return np.log(p / (1.0 - p))

    def compute_probs_gh(self, target_class=None):
        # Mean and the variance of the function corresponding to the observed class
        input_means, input_vars = self.input_means, self.input_vars + tf.exp(self.lvar_noise)
        S, N = tf.shape(input_means)[0], tf.shape(input_means)[1]

        if target_class is not None:
            targets = tf.ones(shape=(N,), dtype=config.int_type_tf) * target_class  # (N,)
        else:
            targets = tf.reshape(self.test_targets_tf, (-1,))  # (N,)

        # Part of the next code is extracted from GPflow
        # uses gauss hermite quadrature (see wikipedia article)
        # targets = 0, 1, 2, 3, ... K - 1
        gh_x, gh_w = self.hermgauss(self.grid_size)  # (grid_size, )
        gh_w = tf.tile(gh_w[None, :, None], [S, 1, 1])  # (S, grid_size, 1)

        # Targets expressed in a one hot enconding matrix with ones in the position of the class
        targets_one_hot_on = tf.one_hot(targets, self.n_classes, tf.constant(1.0, config.float_type_tf), tf.constant(0.0, config.float_type_tf), dtype=config.float_type_tf)  # (S, N, K)
        
        # Only select the means or vars corresponding to the dimension
        #  of the class that we are interested (reduce over K)
        means_class_y_selected = tf.reduce_sum(targets_one_hot_on * input_means, -1, keepdims=True)  # (S, N, 1)
        vars_class_y_selected = tf.reduce_sum(targets_one_hot_on * input_vars, -1, keepdims=True)  # (S, N, 1)

        # As we have to do a change of variable for the Gauss-Hermite quadrature
        # we calculate all the points to evaluate the quadrature.
        X = means_class_y_selected + gh_x * tf.sqrt(tf.clip_by_value(2.0 * vars_class_y_selected, 1e-10, np.inf))  # (S, N, grid_size)

        dist = (tf.expand_dims(X, 2) - tf.expand_dims(input_means, 3)) / tf.expand_dims(tf.sqrt(tf.clip_by_value(input_vars, 1e-10, np.inf)), 3)  # (S, N, K, grid_size)
    
        cdfs = self.norm.cdf(dist)  # (S, N, K, grid_size)

        # One in positions different to targets (logical not of targets_one_hot_on)
        oh_off = tf.cast(tf.one_hot(targets, self.n_classes, 0.0, 1.0), config.float_type_tf)
        oh_off_tiled_by_samples = tf.tile(oh_off[None, :, :], [S, 1, 1])

        # Blank out all the distances on the selected latent function
        cdfs = cdfs * (1 - 2e-4) + 1e-4
        cdfs = cdfs * tf.expand_dims(oh_off_tiled_by_samples, 3) + tf.expand_dims(tf.tile(targets_one_hot_on[None, :, :], [S, 1, 1]), 3)

        # Reduce over the classes. (product of k not equal to y_i)
        cdfs_reduced = tf.reduce_prod(cdfs, 2)

        # reduce_mean over samples
        probs = tf.reduce_mean(cdfs_reduced @ gh_w / tf.sqrt(tf.constant(np.pi, dtype=config.float_type_tf)), 0)  # Final result -> (N, 1)

        return probs

    def hermgauss(self, n: int):
        # This has been extracted from GP flow. Return the locations and weights of GH quadrature
        x, w = np.polynomial.hermite.hermgauss(n)
        return x.astype(config.float_type_np), w.astype(config.float_type_np)

    def getLayerContributionToEnergy(self):
        '''
        Computes logZi, the contribution of the output layer to the energy to optimize
        It computes it by gauss-hermite quadrature

        '''
        probs = self.compute_probs_gh(tf.cast(self.training_targets[:,0], dtype=config.int_type_tf))

        eps = tf.sigmoid(self.latent_prob_wrong_label)
        log_Z = tf.log(probs * (1 - eps)**self.alpha + (1 - probs) * (eps / (self.n_classes - 1))**self.alpha)
        log_Z = tf.reduce_sum(log_Z)
        log_Z /= self.alpha

        return log_Z

    def getPredictedValues(self):

        probs = []
        for target_class in range(self.n_classes):
            probs.append(self.compute_probs_gh(target_class=target_class))

        # Returns labels and confidence on prediction.
        # sum(probs_i) should be 1
        probs = tf.stack(probs, axis=1)

        # The classification rule is:
        # y_i = arg max_k  f_k(x_i)
        # That means that y_i is assigned the index of the latent function with higher value
        labels = tf.argmax(probs, 1)

        return labels, probs

    def getLogLikelihoodError(self):
        """
        Computes the log likelihood and the error rate for a test set
        """

        labels, prob = self.getPredictedValues()

        sq_diff = tf.cast(tf.not_equal(labels, self.test_targets_tf), dtype=config.float_type_tf)

        targets_one_hot_on = tf.one_hot(self.test_targets_tf[:,0], self.n_classes, tf.constant(1.0, config.float_type_tf), tf.constant(0.0, config.float_type_tf), dtype=config.float_type_tf)
        probs_y_selected = tf.reduce_sum(tf.squeeze(prob) * targets_one_hot_on, 1)
        eps = tf.sigmoid(self.latent_prob_wrong_label)
        probs_y_selected = probs_y_selected * (1 - eps) + (1 - probs_y_selected) * (eps / (self.n_classes - 1))

        return tf.log(probs_y_selected), sq_diff

    def sampleFromPredictive(self):
        """
        Sample from the predictive distribution of the deep GP
        """
        pass

    def sampleFromPosterior(self):
        """
        Sample from the posterior distribution of the deep GP
        """
        pass

    def getPredictiveDistribution(self):
        """
        """
        pass
