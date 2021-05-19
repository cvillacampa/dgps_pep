#!/usr/bin/env python3

import sys
sys.path.append('../..')
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#warnings.filterwarnings("ignore")

import numpy as np
import time
from scipy.cluster.vq import kmeans2

from tensorflow.train import AdamOptimizer
from dataset_loader import DatasetLoader
from dgps_pepmcm.gp_network import GPNetwork


def load_dataset(dataset_name, split=0):
    dataset = DatasetLoader(dataset_name, '../../datasets/')

    data = dataset.get_data(split=split)
    X, Y, Xs, Ys, Y_mean, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_mean', 'Y_std']]

    return [X, Y, Xs, Ys, Y_mean, Y_std]


def build_model(_seed, n_hiddens, nolayers, n_samples, M, alpha, var_noise, likelihood_var, shared_z, X_train=None, y_train=None):

    warnings.filterwarnings('error')

    empty_cluster = True
    while (empty_cluster):
        try:
            Z = kmeans2(X_train, M, minit='points')[0]
            empty_cluster = False
        except Warning:
            pass

    warnings.resetwarnings()

    # We construct the network

    if not n_hiddens:
        n_hiddens = min(30, X_train.shape[1])

    net = GPNetwork(X_train, y_train, nolayers, n_hiddens, M, no_samples=n_samples, inducing_points=Z, var_noise=var_noise, lik_noise=likelihood_var, shared_prior=shared_z, alpha=alpha, seed=_seed)

    return net


def train(_seed, minibatch_size, no_iterations, lrate, show_training_info, net=None, n_data=1e-4):
    if not minibatch_size:
        minibatch_size = min(1e4, n_data)

    n_batches = np.ceil(n_data / minibatch_size)
    n_epochs = int(np.ceil(no_iterations / n_batches))

    # train
    t0 = time.process_time()
    net.train(AdamOptimizer(lrate), n_epochs, minibatch_size=minibatch_size, show_training_info=show_training_info)
    t1 = time.process_time()

    return (t1-t0)


def main():
    
    if (len(sys.argv) != 5):
        print("Usage: \n\tpython3 run_uci.py <dataset_name> <split> <n_layers> <alpha>")
        exit(-1)

    model = {
        'n_hiddens': None,
        'nolayers': int(sys.argv[3]),  # 2,3,4,5
        'n_samples': 20,
        'M': 100,
        'alpha': float(sys.argv[4]),
        'var_noise': 1e-5,
        'likelihood_var': 0.01,
        'shared_z': False
    }
    show_training_info = True
    dataset_name = sys.argv[1]
    fixed_split = int(sys.argv[2])
    no_iterations = 20000
    minibatch_size = 100  # min(10k, N)
    lrate = 0.001
    seed = 0

    likelihoods, rmses, times = [], [], []

    print("[INFO] - Training model on split %d" % fixed_split)

    [X_train, y_train, X_test, y_test, y_train_mean, y_train_std] = load_dataset(dataset_name, split=fixed_split)

    net = build_model(seed, model['n_hiddens'], model['nolayers'], model['n_samples'], model['M'], model['alpha'], model['var_noise'], model['likelihood_var'], model['shared_z'], X_train=X_train, y_train=y_train)

    training_time = train(seed, minibatch_size, no_iterations, lrate, show_training_info, net=net, n_data=X_train.shape[0])

    lik, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)

    likelihoods.append(lik)
    rmses.append(rmse)
    times.append(training_time)

    print("[INFO] - Current LL: {} and rmse: {} at split {}".format(lik, rmse, fixed_split))



if __name__ == "__main__":
    main()
