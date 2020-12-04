import tensorflow as tf

from dgps_pepmcm.layers.base_layer import BaseLayer
import dgps_pepmcm.config as config


class NoisyLayer(BaseLayer):
    """
    Class that represents a Noisy Layer
    """

    def __init__(self, var_noise, no_nodes, input_means, input_vars, id):
        """
        Instantiates a noisy layer for the network
            :param var_noise: Variance of the noise
            :param input_means: Output means of the previous layer
            :param input_vars: Output vars of the previous layer
        """
        BaseLayer.__init__(self)

        self.no_nodes = no_nodes
        self.input_means = input_means
        self.input_vars = input_vars

        init_value = tf.math.log(tf.constant(var_noise, dtype=config.float_type_tf)) * tf.ones([1, 1, self.no_nodes], dtype=config.float_type_tf)
        with tf.compat.v1.variable_scope(f"Layer_{id}_Noisy"):
            self.lvar_noise = tf.compat.v1.get_variable('lvar_noise', initializer=init_value, dtype=config.float_type_tf)

    def getOutput(self):
        """
        Sets the output of the noisy layer
        """
        self.output_means = self.input_means
        self.output_vars = self.input_vars + tf.exp(self.lvar_noise)

        return self.output_means, self.output_vars
