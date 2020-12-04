#!/usr/bin/env python3

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import warnings
from scipy.cluster.vq import kmeans2
# Sacred imports
from sacred import Experiment
from tensorflow.train import AdamOptimizer
import sys
sys.path.append('../..')
from dataset_loader import DatasetLoader
from dgps_pepmcm.gp_network import GPNetwork


ex = Experiment()  # Sacred experiment

@ex.config
def my_config():
    model = {
        'n_hiddens': None,
        'nolayers': 2,  # 2,3,4,5
        'n_samples': 20,
        'M': 100,
        'alpha': 1.0,
        'var_noise': 1e-5,
        'likelihood_var': 0.01,
        'shared_z': False
    }
    show_training_info = True
    dataset_name = "boston"
    num_splits = 20
    fixed_split = None # None
    start_split = None
    no_iterations = 20000
    minibatch_size = 100  # min(10k, N)
    lrate = 0.001


@ex.capture
def load_dataset(dataset_name, split=0):
    dataset = DatasetLoader(dataset_name, 'regression', '../../datasets/')

    data = dataset.get_data(split=split)
    X, Y, Xs, Ys, Y_mean, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_mean', 'Y_std']]

    return [X, Y, Xs, Ys, Y_mean, Y_std]


@ex.capture(prefix='model')
def build_model(_seed, n_hiddens, nolayers, n_samples, M, alpha, var_noise, likelihood_var, shared_z, _run, X_train=None, y_train=None):

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

    net = GPNetwork(X_train, y_train, nolayers, n_hiddens, M, no_samples=n_samples, inducing_points=Z, var_noise=var_noise, lik_noise=likelihood_var, shared_prior=shared_z, alpha=alpha, seed=_seed, sacred_exp=_run)

    return net


@ex.capture
def log_results(_log, split=None, lik=None, rmse=None):
    _log.info("Current LL: {} and rmse: {} at split {}".format(lik, rmse, split))


@ex.capture
def write_results(dataset_name, model, _log, split=0, training_time=None, lik=None, rmse=None, dimension=30):
    if not model['n_hiddens']:
        n_hiddens = min(30, dimension)
    else:
        n_hiddens = model['n_hiddens']

    # prepare output files
    outname1 = './results/' + dataset_name + '_' + str(split) + '_' + str(n_hiddens) + '_' + str(model['M']) + '.rmse'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = './results/' + dataset_name + '_' + str(split) + '_' + str(n_hiddens) + '_' + str(model['M']) + '.nll'
    outfile2 = open(outname2, 'w')
    outname3 = './results/' + dataset_name + '_' + str(split) + '_' + str(n_hiddens) + '_' + str(model['M']) + '.time'
    outfile3 = open(outname3, 'w')

    # Add the artifacts for the experiment

    ex.add_artifact(outname1, name='RMSE')
    ex.add_artifact(outname2, name='NLL')
    ex.add_artifact(outname2, name='Time')

    outfile3.write('%.6f\n' % (training_time))
    outfile3.flush()
    os.fsync(outfile3.fileno())

    # We compute the test RMSE and test log-likelihood

    outfile1.write('%.6f\n' % rmse)
    outfile1.flush()
    os.fsync(outfile1.fileno())
    outfile2.write('%.6f\n' % lik)
    outfile2.flush()
    os.fsync(outfile2.fileno())

    outfile1.close()
    outfile2.close()
    outfile3.close()

    return [outfile1, outfile2, outfile3]


@ex.capture
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


@ex.automain
# @LogFileWriter(ex)
def launch_experiment(_run, num_splits, fixed_split, start_split, dataset_name, _log):

    likelihoods, rmses, times = [], [], []

    if fixed_split is not None:
        _log.info("Training model on split %d" % fixed_split)
        iterset = [fixed_split]
    else:
        if start_split is not None:
            _log.info("Training model from split %d to %d" % (start_split, num_splits))
            iterset = range(start_split, num_splits)
        else:
            _log.info("Training model for all the %d splits" % (num_splits))
            iterset = range(num_splits)

    for i in iterset:
        [X_train, y_train, X_test, y_test, y_train_mean, y_train_std] = load_dataset(split=i)

        net = build_model(X_train=X_train, y_train=y_train)

        training_time = train(net=net, n_data=X_train.shape[0])

        lik, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)

        likelihoods.append(lik)
        rmses.append(rmse)
        times.append(training_time)

        # write_results(split = i, training_time=training_time, lik=lik, rmse=rmse, dimension=X_train.shape[1])

        log_results(lik=lik, rmse=rmse, split=i)

        _run.log_scalar("splits.ll", lik)
        _run.log_scalar("splits.rmse", rmse)
        _run.log_scalar("splits.time", training_time)

    testll = np.mean(likelihoods)
    rmse = np.mean(rmses)
    time = np.mean(times)
    print("Average likelihood {}".format(testll))
    print("Average rmse {}".format(rmse))

    _run.info.update({
        "test_rmse": rmse,
        "test_ll": testll,
        "rmse_std": np.std(rmses),
        "ll_std": np.std(likelihoods)
    })

    return "%f, %f, %f" % (testll, rmse, time)
