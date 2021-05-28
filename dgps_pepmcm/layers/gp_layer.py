##
# This file contains the object that represents an input layer to the network
#
import numpy as np

import tensorflow as tf

import dgps_pepmcm.config as config
from dgps_pepmcm.layers import BaseLayer

from dgps_pepmcm.mean.linear import mean_function


class GPLayer(BaseLayer):
    """
    Class representing a GP layer in the network
    """

    def __init__(self, no_inducing_points, no_points, no_nodes, input_d,
                 no_samples, shared_prior, set_for_training, alpha, seed, initialization_dict,
                 input_means, input_vars, id, kernel):
        """
        Instantiates a GP node
            :param int no_inducing_points: Number of inducing points to use on the node (M)
            :param int no_points: Total number of training points
            :param int no_nodes: Number of nodes to add in this GP Layer
            :param int input_d: Input dimensions
            :param int no_samples: Number of samples to use when sampling from the normal dist
            :param shared_prior: Indicates if the prior is shared between the nodes of the layer
            :param float set_for_training: 1.0 if training, 0.0 if prediction (changes cavity)
            :param float alpha: Parameter alpha for alpha-divergence minimization
            :param dictionary initialization_dict: Dictionary containint the initialization for the parameters
                {'W', 'inducing_points', 'lls', 'lsf'}
            :param input_means: Output means of the previous layer
            :param input_vars: Output vars of the previous layer
        """
        assert no_points > 0 and no_nodes > 0 and input_d > 0

        BaseLayer.__init__(self)
   
        self.no_nodes = no_nodes
        self.no_inducing_points = no_inducing_points
        self.no_points = no_points
        self.input_d = input_d
        self.no_samples = no_samples
        self.shared_prior = shared_prior
        self.input_means = input_means
        self.input_vars = input_vars
        self.alpha = alpha
        self.set_for_training = set_for_training
        self.W = initialization_dict['W']
        self.seed = seed
        self.kernel_type = kernel

        with tf.compat.v1.variable_scope(f"Layer_{id}_GP"):
            self.lls = tf.compat.v1.get_variable('lls', initializer=initialization_dict['Lengthscales'], dtype=config.float_type_tf)
            self.lsf = tf.compat.v1.get_variable('lsf', initializer=initialization_dict['lsf'], dtype=config.float_type_tf)
            self.inducing_points = tf.compat.v1.get_variable('z', initializer=initialization_dict['inducing_points'], dtype=config.float_type_tf)
            self.LParamPost = tf.compat.v1.get_variable('LParamPost', initializer=initialization_dict['LParamPost'], dtype=config.float_type_tf)
            self.mParamPost = tf.compat.v1.get_variable('mParamPost', initializer=initialization_dict['mParamPost'], dtype=config.float_type_tf)

    def getOutputNode(self, node_id):
        """
        Computes the output means and variances of a node in the GP Layer

            :param node_id: The id of the node for which we are computing the output
        """
        # We sample from a normal dist. with the input data to the node as param.
        input_samples = tf.random.normal(tf.shape(self.input_means), dtype=config.float_type_tf, seed=self.seed)
        # self.input_samples = tf.ones(tf.shape(self.input_means), dtype=config.float_type_tf)

        input_samples = self.input_means + input_samples * (self.input_vars) ** 0.5

        S, N, D, M = tf.shape(input_samples)[0], tf.shape(input_samples)[1], self.input_d, self.no_inducing_points
        
        # Compute the kernel matrix
        if self.kernel_type=="gauss":
            from dgps_pepmcm.kernel.gauss import compute_kernel
        elif self.kernel_type=="poly":
            from dgps_pepmcm.kernel.polynomial import compute_kernel

        input_samples_flat = tf.reshape(input_samples, [S*N, D])
        # The kernel returns shape (S*N, M) and we convert it to the correct (S,N,M)
        # TODO: modify kernel to accept a (S,N,D) and (M,D) instead of having to reshape
        Kxz = tf.reshape(compute_kernel(self.lls[node_id], self.lsf[node_id], input_samples_flat, self.inducing_points[node_id]), [S, N, M])

        # The kernel returns (M,M) and we want (S,M,M)
        # Tensorflow doesn't have tile minimization implemented
        # self.Kzz = tf.tile(compute_kernel(self.lls, self.lsf, self.inducing_points)[None,:,:], [S, 1, 1])
        Kzz = compute_kernel(self.lls[node_id], self.lsf[node_id], self.inducing_points[node_id])

        chol_Kzz = tf.linalg.cholesky(Kzz, name='kzzChol')
        # Ver como hacer la inversion estable haciendo la inversion de Kzz S veces (S,M,M)
        # (M,M)
        KzzInv = tf.linalg.cholesky_solve(chol_Kzz, tf.eye(M, dtype=config.float_type_tf), name='KzzInv')  # * tf.ones([S,M,M])
        # self.KzzInv = tf.matrix_inverse(self.Kzz, name="KzzInv")
        # (M,M)
        LParamPost_tri = tf.linalg.band_part(self.LParamPost[node_id], -1, 0, name='UTriang_LParamPost') - tf.linalg.tensor_diag(tf.linalg.tensor_diag_part(self.LParamPost[node_id])) + tf.linalg.tensor_diag(tf.exp(tf.linalg.tensor_diag_part(self.LParamPost[node_id])))

        # (M,M)
        LtL = tf.matmul(LParamPost_tri, LParamPost_tri, transpose_b=True, name='LtL')

        mean_prior_z = mean_function(self.inducing_points[node_id], self.W)[:, node_id:node_id+1]
        KzzInvmeanPrior = tf.matmul(KzzInv, mean_prior_z, name='KzzInvmeanPrior')

        # (M,M)
        covCavityInv = KzzInv + LtL * (self.no_points - self.alpha * self.set_for_training) / np.array(self.no_points)
        # Cambiado
        # (M,M)
        covCavity = tf.linalg.cholesky_solve(tf.linalg.cholesky(covCavityInv, name="covCavity"), tf.eye(M, dtype=config.float_type_tf))
        # self.covCavity = tf.matrix_inverse(self.covCavityInv, name="covCavity")
        # self.mParamPost = tf.tile(self.mParamPost[None,:,:], [S,1,1])
        meanCavity = tf.matmul(covCavity, KzzInvmeanPrior + self.mParamPost[node_id] * (self.no_points - self.alpha * self.set_for_training) / np.array(self.no_points))

        KzzInvcovCavity = tf.linalg.cholesky_solve(chol_Kzz, covCavity, name='KzzInvcovCavity')
        KzzInvmean = tf.matmul(KzzInv, (meanCavity - mean_prior_z), name='KzzInvmean')  # A

        mean_term = tf.matmul(Kxz, KzzInvmean * tf.ones([S, M, 1], dtype=config.float_type_tf), name='mean_term')
        mean_prior_x = mean_function(input_samples, tf.tile(self.W[None, :, :], [S, 1, 1]))[:, :, node_id:node_id+1]
        output_means = mean_term + mean_prior_x

        # (M,M)
        covPosteriorInv = KzzInv + LtL

        covPosterior = tf.linalg.cholesky_solve(tf.linalg.cholesky(covPosteriorInv), tf.eye(M, dtype=config.float_type_tf), name='covPosterior')
        # (M,1)
        meanPosterior = tf.matmul(covPosterior, self.mParamPost[node_id] + KzzInvmeanPrior, name='meanPosterior')
        # meanPosterior = tf.matmul(covPosterior, tf.linalg.cholesky_solve(LParamPost_tri, self.mParamPost[node_id]) + KzzInvmeanPrior, name='meanPosterior')
        # meanPosterior = tf.matmul(covPosterior, self.mParamPost[node_id], name='meanPosterior')

        # Compute the output vars
        # (S, N, M)
        B = tf.matmul(KzzInvcovCavity, KzzInv) - KzzInv * tf.ones([S, M, M], dtype=config.float_type_tf)
        # config.jitter + tf.exp(self.lsf) viene del kernel kxx
        v_out = tf.exp(self.lsf[node_id]) + \
            tf.matmul(Kxz * tf.matmul(Kxz, B, name='kzzB'),
                      tf.ones([S, M, 1], dtype=config.float_type_tf), name="matMulVout")

        # Variance of the dist that we want to sample from
        # Size (S,N,1)

        output_vars = tf.abs(v_out)

        # Also return the log normalizers needed to compute the energy

        logNormalizerPrior = 0.5 * self.no_inducing_points * np.log(2 * np.pi) + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_Kzz))) + \
            tf.squeeze(0.5 * tf.matmul(tf.matmul(mean_prior_z, KzzInv, transpose_a=True), mean_prior_z))
        logNormalizerCavity = (0.5 * self.no_inducing_points * np.log(2 * np.pi) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(covCavityInv))))) + \
            tf.squeeze(0.5 * tf.matmul(tf.matmul(meanCavity, covCavityInv, transpose_a=True), meanCavity))
        logNormalizerPosterior = (0.5 * self.no_inducing_points * np.log(2 * np.pi) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(covPosteriorInv))))) + \
            tf.squeeze(0.5 * tf.matmul(tf.matmul(meanPosterior, covPosteriorInv, transpose_a=True), meanPosterior))

        return [output_means, output_vars, logNormalizerPrior, logNormalizerCavity, logNormalizerPosterior]

    def getOutput(self):
        """
        Computes the output of the GP Layer by iterating over the number of nodes.
        It computes:
            - output_means: The output means of the GP layer
            - output_vars: The output variances of the GP layer
            - logNormalizerPrior: The log normalizer of the prior
            - logNormalizerCavity: The log normalizer of the cavity
            - logNormalizerPosterior: The log normalizer of the posterior
        """

        output_means = []
        output_vars = []
        logNormalizerPrior = []
        logNormalizerCavity = []
        logNormalizerPosterior = []

        if self.shared_prior:
            iterset = np.zeros(self.no_nodes).astype(config.int_type_np)
        else:
            iterset = range(self.no_nodes)

        # Iterate over the nodes
        for i in iterset:
            [outmeans, outvars, logZPrior, logZCavity, logZPosterior] = self.getOutputNode(i)
            output_means += [outmeans]
            output_vars += [outvars]
            logNormalizerPrior += [logZPrior]
            logNormalizerCavity += [logZCavity]
            logNormalizerPosterior += [logZPosterior]

        # We group the results of the individual nodes into a single tensor
        self.output_means = tf.concat(output_means, axis=2)
        self.output_vars = tf.concat(output_vars, axis=2)
        self.logNormalizerPrior = tf.stack(logNormalizerPrior, axis=0)
        self.logNormalizerCavity = tf.stack(logNormalizerCavity, axis=0)
        self.logNormalizerPosterior = tf.stack(logNormalizerPosterior, axis=0)

        return self.output_means, self.output_vars

    def getLayerContributionToEnergy(self):
        """
        We return the contribution to the energy of the layer
        (See last Eq. of Sec. 4 in http://arxiv.org/pdf/1602.04133.pdf v1)
        """
        n_minibatch = tf.cast(tf.shape(self.input_means)[1], config.float_type_tf)

        # We multiply by the minibatch size and normalize terms according to the total number of points (no_points)
        contribution = (self.logNormalizerCavity - self.logNormalizerPosterior) / self.alpha
        contribution += self.logNormalizerPosterior / self.no_points - self.logNormalizerPrior / self.no_points
        contribution *= n_minibatch

        # return contribution
        return tf.reduce_sum(contribution)
