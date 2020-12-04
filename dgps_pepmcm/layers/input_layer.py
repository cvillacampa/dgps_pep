##
# This file contains the object that represents an input layer to the network
#
import tensorflow as tf

import dgps_pepmcm.config as config
from dgps_pepmcm.layers.base_layer import BaseLayer


class InputLayer(BaseLayer):
    """
    Class representing an input layer in the network
    """

    def __init__(self, training_data, d_input, n_samples, set_for_training):
        """Instantiates an input layer for the network. It has one node that outputs (S,N,D)

            :param training_data: Input data for the network.
            :param d_input: Number of dimensions of the input data.
            :param n_samples: Number of samples to propagate trough the network.
            :param set_for_training: 1.0 for training 0.0 for prediction

        """
        BaseLayer.__init__(self)

        self.no_nodes = d_input
        self.no_samples = n_samples
        self.training_data = training_data
        self.set_for_training = set_for_training

    def output_training(self):
        """
        Output of the input layer when we are in training "mode"
        """
        return tf.tile(self.training_data[None, :, :], [self.no_samples['training'], 1, 1])

    def output_prediction(self):
        """
        Output of the input layer when we are in prediction "mode"
        """
        return tf.tile(self.training_data[None, :, :], [self.no_samples['prediction'], 1, 1])

    def getOutput(self):
        """
        Computes the output for an input node which is a tensor
        with the training data replicated n_samples(S) times.
        The shape of this tensor is (S,N,D).
        """
        self.output_means = tf.cond(tf.equal(self.set_for_training, tf.constant(1.0, dtype=config.float_type_tf)), self.output_training, self.output_prediction)
        self.output_vars = self.output_means * 0.0

        return self.output_means, self.output_vars
