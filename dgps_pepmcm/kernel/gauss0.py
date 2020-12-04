import dgps_pepmcm.config as config
import numpy as np
import tensorflow as tf


class SquaredExponential():
    """
    Class representing a Squared Exponential kernel
    """

    def __init__(self, lls, lsf):
        """
        Instantiates a squared exponential kernel
        """
        self.lls = lls
        self.lsf = lsf

    def compute_kernel_tensor(self, X, X2 = None):
        """
        This function computes the covariance matrix for the GP
        """
        if X2 is None:
            X2 = X
            jitter = 1e-5
            white_noise = jitter * tf.eye(tf.shape(X)[1], dtype = config.float_type_tf)
        else:
            white_noise = 0.0

        lls = tf.tile(lls[:,None,:],[1,tf.shape(X)[1],1])
        lsf = tf.expand_dims(tf.expand_dims(lsf,1),2)
        # lsf = tf.tile(tf.expand_dims(lsf,1)[:,None,:],[1,tf.shape(X)[1],1])

        X = X / tf.sqrt(tf.exp(lls))
        X2 = X2 / tf.sqrt(tf.exp(lls))
        value = tf.tile(tf.reduce_sum(tf.square(X), 1)[:,None,:], [1,tf.shape(X)[1],1])
        value2 = tf.tile(tf.reduce_sum(tf.square(X2), 1)[:,None,:], [1,tf.shape(X2)[1],1])
        # value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        # value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)

        distance = value - 2 * tf.matmul(X, X2, transpose_b=True) + tf.transpose(value2, [0,2,1])

        return tf.exp(lsf) * tf.exp(-0.5 * distance) + white_noise


    def compute_kernel(lls, lsf, X, X2=None):
        """
        This function computes the covariance matrix for the GP
        """
        if X2 is None:
            X2 = X
            jitter = 1e-5
            white_noise = jitter * tf.eye(tf.shape(X)[0], dtype = config.float_type_tf)
        else:
            white_noise = 0.0

        X = X / tf.sqrt(tf.exp(lls))
        X2 = X2 / tf.sqrt(tf.exp(lls))
        value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)
        distance = value - 2 * tf.matmul(X, X2, transpose_b=True) + tf.transpose(value2)

        return tf.exp(lsf) * tf.exp(-0.5 * distance) + white_noise
