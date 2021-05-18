import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import dgps_pepmcm.config as config
from dgps_pepmcm.layers import BaseLayer



class OutputLayerClassification(BaseLayer):
    """
    This file contains the object that represents an classification output layer to the network
    """

    def __init__(self, training_targets, no_samples, alpha, seed, lik_noise, input_means, input_vars):
        BaseLayer.__init__(self)

        self.norm = tfp.distributions.Normal(loc=tf.constant(0.0, config.float_type_tf), scale=tf.constant(1.0, config.float_type_tf))
        
        self.training_targets = training_targets
        self.no_samples = no_samples
        self.no_nodes = 1
        self.grid_size = 100
        self.alpha = alpha
        self.seed = seed
        self.input_means = input_means
        self.input_vars = input_vars
        self.lvar_noise = tf.compat.v1.get_variable("lik_noise", initializer=tf.math.log(tf.constant(lik_noise, dtype=config.float_type_tf)), dtype=config.float_type_tf)
        self.test_targets_tf = tf.compat.v1.placeholder(config.int_type_tf, name="y_test", shape=[None, 1])

    def getLayerContributionToEnergy(self):
        '''
        Computes logZi, the contribution of the output layer to the energy to optimize

        From Bui et al. 2017 (G.2 Classification):
        this quantity can be evaluated numerically, using sampling or GaussHermite quadrature, since it only involves a one-dimensional integral
        '''
        logZ = self.getLogZ()
        logZ = tf.reduce_sum(logZ)
        logZ /= self.alpha

        return logZ

    def getLogZ(self, targets=None):
        """
        Compute logZ using gauss-hermite quadrature
        """

        input_means, input_vars = self.input_means, self.input_vars + tf.exp(self.lvar_noise)
        S = tf.shape(input_means)[0]

        if targets is None:
            targets = self.training_targets

        gh_x, gh_w = self.hermgauss(self.grid_size)

        gh_w = gh_w[None, None, :]
        ts = gh_x * tf.sqrt(2.0 * input_vars) + input_means

        log_cdfs = self.norm.log_cdf(targets*ts)

        logZ = tf.reduce_logsumexp(self.alpha*log_cdfs + tf.math.log(gh_w), axis=2) - 0.5 * tf.math.log(tf.constant(np.pi, dtype=config.float_type_tf))

        return tf.reduce_logsumexp(logZ, 0) - tf.math.log(tf.cast(S, config.float_type_tf))

    def hermgauss(self, n: int):
        # This has been extracted from GP flow. Return the locations and weights of GH quadrature
        x, w = np.polynomial.hermite.hermgauss(n)
        return x.astype(config.float_type_np), w.astype(config.float_type_np)

    def getPredictedValues(self):
        input_means, input_vars = self.input_means, self.input_vars  # S, N, 1
        S = tf.shape(input_means)[0]

        # (S, N, 1)
        alpha = input_means / tf.sqrt(1 + input_vars + tf.exp(self.lvar_noise))
        # (N, 1)
        prob = tf.exp(
            tf.reduce_logsumexp(self.norm.log_cdf(alpha), 0)
            - tf.math.log(tf.cast(S, config.float_type_tf))
        )

        # label[n] = -1 if input_means[n] < 0  else 1
        labels = tf.where(
            tf.less(tf.reduce_sum(input_means, 0), tf.zeros_like(prob)),
            -1 * tf.ones(tf.shape(prob), dtype=config.int_type_tf),
            tf.ones(tf.shape(prob), dtype=config.int_type_tf),
        )
    
        return labels, prob

    def calculate_log_likelihood(self):
        # The only difference with the function above is that
        # the means and vars should be calculated using the posterior instead of the cavity
        # and y_test should be used.
        input_means, input_vars = self.input_means, self.input_vars  # S, N, 1
        S = tf.shape(input_means)[0]

        # Parameter of the log cdf
        alpha = (
            tf.cast(self.test_targets_tf, config.float_type_tf)
            * input_means
            / tf.sqrt(1 + input_vars + tf.exp(self.lvar_noise))
        )

        return tf.reduce_logsumexp(self.norm.log_cdf(alpha), 0) - tf.math.log(
            tf.cast(S, config.float_type_tf)
        )

    def sample_from_latent(self):
        input_means, input_vars = self.input_means, self.input_vars  # S, N, 1
        # Returns samples from H^L
        return tf.random_normal(
            tf.shape(input_means),
            mean=input_means,
            stddev=tf.sqrt(input_vars),
            seed=self.seed,
            dtype=config.float_type_tf
        )

    def getLogLikelihoodError(self):
        """
        Computes the log likelihood and the error rate for a test set
        """

        labels, _ = self.getPredictedValues()
        err_rate = tf.cast(tf.not_equal(labels, self.test_targets_tf), dtype=config.float_type_tf)

        probs_y_selected = self.calculate_log_likelihood()

        return probs_y_selected, err_rate

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