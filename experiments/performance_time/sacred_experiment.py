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
        'n_hiddens': 8,
        'nolayers': 2,  # 2,3,4,5
        'n_samples': 20,
        'M': 100,
        'alpha': 1.0,
        'var_noise': 1e-5,
        'likelihood_var': 0.01,
        'shared_z': False
    }
    show_training_info = False
    dataset_name = "boston"
    split = 0
    no_iterations = 200000
    minibatch_size = 100  # min(10k, N)
    lrate = 0.01


@ex.capture
def load_dataset(dataset_name, split):
    if dataset_name in ["higgs", "airlines_classification"]:
        dataset_type = "classification"
        from_database = True
    elif dataset_name == "mnist":
        dataset_type = "multiclass"
        from_database = True
    else:
        dataset_type = "regression"
        from_database = True

    dataset = DatasetLoader(dataset_name, dataset_type, '../../datasets/', from_database)
    data = dataset.get_data(split=split)

    if dataset_type == "regression":
        X, Y, Xs, Ys, Y_mean, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_mean', 'Y_std']]
        return [X, Y, Xs, Ys, Y_mean, Y_std]
    else:
        X, Y, Xs, Ys = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys']]
        return [X, Y, Xs, Ys, np.array([0.0]), np.array([1.0])]


@ex.capture(prefix='model')
def build_model(_seed, n_hiddens, nolayers, n_samples, M, alpha, var_noise, likelihood_var, shared_z, _run, X_train=None, y_train=None):

    warnings.filterwarnings('error')

    empty_cluster = True
    i = 0
    while (empty_cluster):
        try:
            Z = kmeans2(X_train, M, minit='points')[0]
            empty_cluster = False
        except Warning:
            pass
        finally:
            i+=1

    warnings.resetwarnings()

    if empty_cluster:
        Z = np.random.uniform(low=-1., high=1., size=(M, X_train.shape[1]))

    # We construct the network

    if not n_hiddens:
        n_hiddens = min(30, X_train.shape[1])

    net = GPNetwork(X_train, y_train, nolayers, n_hiddens, M, no_samples=n_samples, inducing_points=Z, var_noise=var_noise, lik_noise=likelihood_var,
                        shared_prior=shared_z, alpha=alpha, seed=_seed, sacred_exp=_run)

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
def train(_seed, minibatch_size, no_iterations, lrate, show_training_info, net=None, n_data=1e-4, X_test=None, y_test=None, y_train_std=None, y_train_mean=None,):
    if not minibatch_size:
        minibatch_size = min(1e4, n_data)

    n_batches = np.ceil(n_data / minibatch_size)
    n_epochs = int(np.ceil(no_iterations / n_batches))

    # train
    t0 = time.process_time()
    net.train(AdamOptimizer(lrate), n_epochs, minibatch_size=minibatch_size, X_test=X_test,
              show_training_info=show_training_info, y_test=y_test, y_train_std=y_train_std,
              y_train_mean=y_train_mean, log_every=100)
    t1 = time.process_time()

    return (t1-t0)


@ex.automain
# @LogFileWriter(ex)
def launch_experiment(_run, split, dataset_name, _log):

    [X_train, y_train, X_test, y_test, y_train_mean, y_train_std] = load_dataset(split=split)

    net = build_model(X_train=X_train, y_train=y_train)

    training_time = train(net=net, n_data=X_train.shape[0], X_test=X_test,
                          y_test=y_test, y_train_std=y_train_std, y_train_mean=y_train_mean)

    lik, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)

    # write_results(split = i, training_time=training_time, lik=lik, rmse=rmse, dimension=X_train.shape[1])

    log_results(lik=lik, rmse=rmse, split=split)

    print("Average likelihood {}".format(lik))
    print("Average rmse {}".format(rmse))

    _run.info.update({
        "test_rmse": rmse,
        "test_ll": lik
    })

    return "%f, %f, %f" % (lik, rmse, training_time)
