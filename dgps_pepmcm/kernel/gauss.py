import dgps_pepmcm.config as config
import tensorflow as tf


def compute_kernel(lls, lsf, X, X2=None):
    """
    This function computes the covariance matrix for the GP
    """
    if X2 is None:
        X2 = X
        white_noise = config.jitter * tf.eye(tf.shape(X)[0], dtype=config.float_type_tf)
    else:
        white_noise = 0.0

    X = X / tf.sqrt(tf.exp(lls))
    X2 = X2 / tf.sqrt(tf.exp(lls))
    value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
    value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)
    distance = value - 2 * tf.matmul(X, X2, transpose_b=True) + tf.transpose(value2)

    return tf.exp(lsf) * tf.exp(-0.5 * distance) + white_noise
