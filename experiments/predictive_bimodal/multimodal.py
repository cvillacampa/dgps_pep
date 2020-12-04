import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scipy.cluster.vq import kmeans2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.train import AdamOptimizer
import sys
sys.path.append('../..')

from dgps_pepmcm.gp_network import GPNetwork


def bimodal1(nopoints):
    x = np.linspace(-2, 2, nopoints).reshape(nopoints, 1)
    prob = np.random.rand(nopoints).reshape(nopoints, 1)
    y = x.copy()
    y[prob < 0.5] = 10 * np.sin(1 * x[prob < 0.5])
    y[prob > 0.5] = 10 * np.cos(1 * x[prob > 0.5])
    y += np.random.randn(nopoints, 1)

    return x, y


def bimodal2(nopoints):
    x = np.linspace(-4, 4, nopoints).reshape(nopoints, 1)
    y = 7 * np.sin(x) + 3 * np.abs(np.cos(x / 2)) * np.random.randn(nopoints, 1)

    return x, y


if len(sys.argv) != 3:
    print("Error: Incorrect format: \n\tpython3 multimodal.py alpha (1|2)")
    sys.exit(1)

alpha = float(sys.argv[1])
problem = int(sys.argv[2])

#########################################
# Model Variables
#########################################
L = 4  # Number of total layers
n_hiddens = 3
max_iter = 500
notrain = 1000
notest = 500
pdf_graph = False  # Paint pdf graph for point
n_samples_each_point = 20  # If sampling for posterior number of samples per point. A high value can cause TF to crash
minibatch_size = 50
n_samples = 50  # Samples to propagate in train
M = 10
if problem == 1:
    learning_rate = 0.01
elif problem == 2:
    learning_rate = 0.002
var_noise = 1e-5
if alpha < 0:
    likelihood_var = 1e-2
else:
    likelihood_var = 1e-2
save_to_file = True  # Save all figures to folder
folder_name = 'figs_multimodal'
seed = 0

#########################################
# Generate data and Normalize
#########################################
np.random.seed(seed)

if problem == 1:
    [X_train, y_train] = bimodal1(notrain)
    [X_test, y_test] = bimodal1(notest)
    if not os.path.exists(f"{folder_name}/problem{problem}/problem.png"):
        os.makedirs(f"{folder_name}/problem{problem}", exist_ok = True)
        data_frame = pd.DataFrame({'x': X_train.flatten(), 'y': y_train.flatten()})
        sns.set_style('white')  # whitegrid darkgrid, white dark ticks
        ax = sns.scatterplot(x='x', y='y', data=data_frame, label="Data points")  # ,palette="colorblind" # deep, muted, pastel, bright, dark, colorblind
        plt.plot(X_train, 10*np.sin(X_train), color="red", linewidth=3, label="Ground truth")
        plt.plot(X_train, 10*np.cos(X_train), color="red", linewidth=3)
        ax.set(xlabel='x', ylabel='y')
        ax.legend(loc='lower right')
        ax.set_title("Training data for bi-modal problem")
        sns.set_context("paper")  # paper poster
        # plt.show()
        plt.savefig(f"{folder_name}/problem{problem}/problem.png")
        plt.close()
elif problem == 2:
    [X_train, y_train] = bimodal2(notrain)
    [X_test, y_test] = bimodal2(notest)
    if not os.path.exists(f"{folder_name}/problem{problem}/problem.png"):
        os.makedirs(f"{folder_name}/problem{problem}", exist_ok = True)
        data_frame = pd.DataFrame({'x': X_train.flatten(), 'y': y_train.flatten()})
        
        sns.set_style('white')  # whitegrid darkgrid, white dark ticks
        ax = sns.scatterplot(x='x', y='y', data=data_frame, label="Data points")  # ,palette="colorblind" # deep, muted, pastel, bright, dark, colorblind
        plt.plot(X_train, 7 * np.sin(X_train), color="red", linewidth=3, label="Ground truth")
        ax.set(xlabel='x', ylabel='y')
        ax.legend(loc='lower right')
        ax.set_title("Training data for heteroskedastic problem")
        sns.set_context("paper")  # paper poster
        plt.savefig(f"{folder_name}/problem{problem}/problem.png")
        plt.close()

X_plot = np.reshape(np.linspace(-1.5, 1.5, notest), (notest, 1))

X_train_mean = np.mean(X_train, 0)[None, :]
X_train_std = np.std(X_train, 0)[None, :] + 1e-6

y_train_mean = np.mean(y_train, 0)
y_train_std = np.std(y_train, 0) + 1e-6

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
X_plot = (X_plot - X_train_mean) / X_train_std

y_train = (y_train - y_train_mean) / y_train_std

#########################################
# Model
#########################################

# Initialize inducing points
Z = kmeans2(X_train, M, minit='points')[0]
net = GPNetwork(X_train, y_train, L, n_hiddens, M, noisy_layers=True, var_noise=var_noise,
                lik_noise=likelihood_var, shared_prior=False, no_samples=n_samples, inducing_points=Z,
                alpha=alpha, seed=seed)
net.train(AdamOptimizer(learning_rate), no_epochs=max_iter, minibatch_size=minibatch_size, show_training_info=True)

testll, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)
print("Test RMSE: {}".format(rmse))
print("Test ll: {}".format(testll))

#########################################
# Plot results
#########################################

if save_to_file and not os.path.exists(f"{folder_name}/problem{problem}"):
    os.makedirs(f"{folder_name}/problem{problem}")

# Sample from predictive
if problem == 1:
    points_x = np.tile(np.linspace(-2.0, 2.0, notest), [n_samples_each_point])[:, None]
elif problem == 2:
    points_x = np.tile(np.linspace(-4.0, 4.0, notest), [n_samples_each_point])[:, None]
points_y = net.sampleFromPredictive((points_x - X_train_mean) / X_train_std, y_train_std, y_train_mean)

if pdf_graph:
    x_value = np.array([-3])[:, None]
    y_range = np.linspace(-1.75, 1.75, num=150)[:, None]
    pdf = net.getPredictiveDist(x_value, y_range)
    plt.figure(2)
    plt.plot(y_range, pdf, 'g-', label="Pred. dist for point {}".format(x_value[0, 0]))
    x_value = np.array([1.75])[:, None]
    pdf = net.getPredictiveDist(x_value, y_range)
    plt.plot(y_range, pdf, 'b-', label="Pred. dist for point {}".format(x_value[0, 0]))
    plt.xlabel('y')
    plt.ylabel('p(x)')
    plt.legend()
    if save_to_file:
        plt.savefig(f"{folder_name}/problem{problem}/alpha{alpha}_M{M}_L{L}_S{n_samples}_mb{minibatch_size}_pdf.png")
        plt.close()
    else:
        plt.show()

plt.figure(1)
data_frame = pd.DataFrame({'x': points_x.flatten(), 'y': points_y.flatten()})
sns.set_style('white')  # whitegrid darkgrid, white dark ticks
ax = sns.scatterplot(x='x', y='y', data=data_frame, color="black", linewidths=0, marker=".")
ax.set(xlabel='x', ylabel='y')
ax.set_ylim(-13, 13)
ax.set_title(f"Predictions alpha {alpha}")
sns.set_context("paper")  # paper poster


if save_to_file:
    print(f"Figures are in folder {folder_name}/problem{problem}")
    plt.savefig(f"{folder_name}/problem{problem}/alpha{alpha}_M{M}_L{L}_S{n_samples}_mb{minibatch_size}.png")
    plt.close()
else:
    plt.show()
