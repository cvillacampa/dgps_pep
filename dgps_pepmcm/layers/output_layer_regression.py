import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import dgps_pepmcm.config as config
from dgps_pepmcm.layers import BaseLayer


class OutputLayerRegression(BaseLayer):
    """
    Class representing an output node for regression problems
    """

    def __init__(self, training_targets, no_samples, alpha, seed, no_points, lik_noise, input_means, input_vars):
        """
        Instantiates an output regression layer

            :param training_targets:
            :param no_samples:
            :param alpha:
            :param seed:
            :param lik_noise: Likelihood noise
            :param input_means
            :param input_vars
        """
        BaseLayer.__init__(self)

        self.training_targets = training_targets
        self.no_samples = no_samples
        self.no_nodes = 1
        self.alpha = alpha
        self.seed = seed
        self.input_means = input_means
        self.input_vars = input_vars
        self.lvar_noise = tf.compat.v1.get_variable("lik_noise", initializer=tf.math.log(tf.constant(lik_noise, dtype=config.float_type_tf)), dtype=config.float_type_tf)
        self.test_targets_tf = tf.compat.v1.placeholder(config.float_type_tf, name="y_test", shape=[None, training_targets.shape[0]])
        self.y_train_std = tf.compat.v1.placeholder(config.float_type_tf, name="y_train_std", shape=[1])
        self.y_train_mean = tf.compat.v1.placeholder(config.float_type_tf, name="y_train_mean", shape=[1])

    def getLayerContributionToEnergy(self):
        '''
        Computes logZi, the contribution of the output layer to the energy to optimize

        From Bui et al. 2017 (G.1 Regression):
        log Z_tilted,n = -\\alpha/2 log(2\\pi\\alpha_y^2 ) + 1/2 log(\\alpha_y^2 ) + 1/2 log(\\alphav_n + \\sigma_y^2 ) + 1/2 (y_n - \\mu_n )^2 / (v_n + \\sigma_y^2 / \\alpha)

        '''
        log_num_samples = tf.math.log(tf.cast(tf.shape(self.input_means)[0], config.float_type_tf))

        logZ = -0.5 * self.alpha * tf.math.log(2 * tf.constant(np.pi, dtype=config.float_type_tf))
        logZ -= 0.5 * self.alpha * self.lvar_noise  # Incluir ruido en la capa de salida
        logZ += 0.5 * self.lvar_noise
        logZ -= 0.5 * tf.math.log(self.alpha * self.input_vars + tf.exp(self.lvar_noise))
        logZ -= 0.5 * tf.square(self.training_targets - self.input_means) / (self.input_vars + tf.exp(self.lvar_noise) / self.alpha)
        logZ = tf.reduce_logsumexp(logZ, 0) - log_num_samples  # Mean over samples
        logZ /= self.alpha

        return tf.reduce_sum(logZ)

    def getPredictedValues(self):
        """
        Get the means and the variances of the predictive distribution
        """
        input_means, input_vars = self.input_means, self.input_vars  # S,N,D
        output_means = tf.reduce_mean(input_means, 0)  # N,D

        # Añadir ruido a la varianza
        input_vars += tf.exp(self.lvar_noise)

        # La varianza de salida es el segundo momento de una mixtura de gaussianas
        output_vars = tf.reduce_mean(output_means**2 + input_vars, 0) - output_means**2

        return output_means, output_vars

    def getLogLikelihoodError(self):
        """
        Computes the log likelihood and the rmse for a test set
        """
        # Expecting not normalized y_test
        [output_means, output_vars] = [self.input_means, self.input_vars + tf.exp(self.lvar_noise)]

        log_num_samples = tf.math.log(tf.cast(tf.shape(output_means)[0], config.float_type_tf))

        norm = tfp.distributions.Normal(
                        loc=output_means * self.y_train_std + self.y_train_mean,
                        scale=tf.sqrt(output_vars) * self.y_train_std)
        logpdf = norm.log_prob(self.test_targets_tf)

        # RMSE
        mean = tf.reduce_mean(output_means, 0) * self.y_train_std + self.y_train_mean  # mean over the samples
        sq_diff = (mean - self.test_targets_tf) ** 2

        # we return the likelihood of each point, we will have to calculate the mean of that
        # or we could do it here in tf with tf.reduce_mean , axis=0
        return (tf.reduce_logsumexp(logpdf, 0) - log_num_samples), sq_diff

    def sampleFromPredictive(self):
        """
        Sample from the predictive distribution of the deep GP
        """

        input_means, input_vars = self.input_means, (self.input_vars + tf.exp(self.lvar_noise))  # S,N,D

        S = tf.shape(input_means, out_type=config.int_type_tf)[0]
        N = tf.shape(input_means, out_type=config.int_type_tf)[1]
        D = tf.shape(input_means, out_type=config.int_type_tf)[2]

        # Sample from mixture of gaussians
        # 1. sample from categorical dist
        probs = tf.cast(1/S, dtype=config.float_type_tf) * tf.ones(S, dtype=config.float_type_tf)
        cat = tfp.distributions.Categorical(probs=probs)
        samples_categorical = tf.cast(cat.sample(sample_shape=tf.cast(N, tf.int32), seed=self.seed), dtype=config.int_type_tf)

        # Get corresponding means and vars
        indexes = tf.concat([samples_categorical[:, None], tf.range(0, N)[:, None]], 1)
        mixture_means = tf.gather_nd(input_means, indexes) * self.y_train_std + self.y_train_mean  # mixture_means = input_means[samples_categorical, tf.range(0, N), :]
        mixture_vars = tf.gather_nd(input_vars, indexes) * self.y_train_std  # Size: N,D

        return tf.random.normal(shape=[N, D], mean=mixture_means, stddev=tf.sqrt(mixture_vars), dtype=config.float_type_tf, seed=self.seed)

    def sampleFromPosterior(self):
        """
        Sample from the posterior distribution of the deep GP
        """

        input_means, input_vars = self.input_means, self.input_vars  # S,N,D
        S = tf.shape(input_means, out_type=config.int_type_tf)[0]
        N = tf.shape(input_means, out_type=config.int_type_tf)[1]
        D = tf.shape(input_means, out_type=config.int_type_tf)[2]

        # Sample from mixture of gaussians
        # 1. sample from categorical dist
        probs = tf.cast(1/S, dtype=config.float_type_tf) * tf.ones(shape=S, dtype=config.float_type_tf)
        cat = tfp.distributions.Categorical(probs=probs)
        samples_categorical = tf.cast(cat.sample(sample_shape=tf.cast(N, tf.int32), seed=self.seed), dtype=config.int_type_tf)

        # Get corresponding means and vars
        indexes = tf.concat([samples_categorical[:, None], tf.range(0, N)[:, None]], 1)
        mixture_means = tf.gather_nd(input_means, indexes) * self.y_train_std + self.y_train_mean  # mixture_means = input_means[samples_categorical, tf.range(0, N), :]
        mixture_vars = tf.gather_nd(input_vars, indexes) * self.y_train_std  # Size: N,D

        return tf.random.normal(shape=[N, D], mean=mixture_means, stddev=tf.sqrt(mixture_vars), dtype=config.float_type_tf, seed=self.seed)

    def getPredictiveDistribution(self):
        """

        """
        norm = tfp.distributions.Normal(loc=self.input_means, scale=tf.sqrt(self.input_vars))

        return tf.reduce_mean(norm.prob(self.training_targets), 0)
