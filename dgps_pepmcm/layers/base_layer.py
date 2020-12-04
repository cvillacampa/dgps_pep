##
# This file contains the base object layer for the layers we may have in the gp network
#

import tensorflow as tf
import dgps_pepmcm.config as config


class BaseLayer:

    """ Base class for all layers, contains common functions used by all layers of the network."""

    def __init__(self):
        # self.node_list = []
        self.no_nodes = None
        self.no_samples = None

    def initialize_params_layer(self):
        """
        This method initializes the parameters of the layer
        Should be overwritten (currently it is only used in Noise_layer and GP_layer)
        """
        pass

    # To be overwritten

    def getOutput(self):
        """
        Should be overwritten to set the output means and vars of the current layer
        """
        pass

    def getLayerContributionToEnergy(self):
        """
        Should be overwritten to return the contribution of the layer to the energy function
        """
        return tf.constant(0.0, dtype=config.float_type_tf)
