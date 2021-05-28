import dgps_pepmcm0.config as config
import tensorflow as tf



def compute_kernel(lls, lsf, X, X2=None, degree=9.0):
    """
    This function computes the covariance matrix for the GP using polynomial kernel
    """
    if X2 is None:
        X2 = X
        white_noise = (config.jitter + tf.exp(lsf)) * tf.eye(tf.shape(X)[ 0 ], dtype = config.float_type_tf)
    else:
        white_noise = 0.0

    X = X / tf.sqrt(tf.exp(lls))
    X2 = X2 / tf.sqrt(tf.exp(lls))
                    
    product = tf.pow((tf.matmul(X, tf.transpose(X2)) + 1.0), degree)
    
    return tf.exp(lsf) * product + white_noise
