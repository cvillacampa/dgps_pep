import numpy as np
from tensorflow.train import AdamOptimizer
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('../..')

from dgps_pepmcm.gp_network import GPNetwork


def step(x):
    y = x.copy()
    y[y < 0.0] = -1.0
    y[y > 0.0] = 1.0
    return y + 0.05 * np.random.randn(x.shape[0], 1)


#########################################
# Model Variables
#########################################

L = 3  # Number of total layers
n_hiddens = 2
max_iter = 1000
notrain = 1000
notest = 500
pdf_graph = False  # Paint pdf graph for point
samples_from_predictive = True
n_samples_each_point = 20  # If sampling for posterior number of samples per point. A high value can cause TF to crash
minibatch_size = 50
n_samples = 50  # Samples to propagate in train
M = 10
learning_rate = 0.01
var_noise = 1e-5
likelihood_var = 1e-2
save_to_file = True  # Save all figures to folder
plot_training = True
folder_name = 'figs_multimodal0'
seed = 1

if len(sys.argv) != 2:
    print("Error: Incorrect format: \n\tpython3 multimodal.py alpha")
    sys.exit(1)

alpha = float(sys.argv[1])

#########################################
# Generate data and Normalize
#########################################
np.random.seed(seed)

X_train = np.reshape(np.linspace(-1, 1, notrain), (notrain, 1))
X_test = np.reshape(np.linspace(-1, 1, notest), (notest, 1))
X_plot = np.reshape(np.linspace(-1.5, 1.5, notest), (notest, 1))

y_train = step(X_train)
y_test = step(X_test)

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
if plot_training:   # Plots the training process
    # First step
    net.train(AdamOptimizer(learning_rate), no_epochs=1, minibatch_size=minibatch_size, show_training_info=False)

    # Prepare dynamic plot
    plt.ion()
    fig = plt.figure(num=2, figsize=(9, 6))
    ax = fig.add_subplot(111)
    points_x = np.tile(np.linspace(-4.0, 4.0, notest), [n_samples_each_point])[:, None]
    line0, = ax.plot(points_x, np.zeros(points_x.shape), 'g.', markersize=2)
    line1, = ax.plot(X_train, y_train, 'bo', alpha=0.5)
    line2, = ax.plot(X_plot, np.zeros(X_test.shape), 'm-')
    line3, = ax.plot(X_plot, np.zeros(X_test.shape), 'm--')
    line4, = ax.plot(X_plot, np.zeros(X_test.shape), 'm--')
    text = ax.text(0.01, 0.90, "RMSE: {:.4f}, LL: {:.4f} \nSamp: {}, iter: {} \nLayers: {} M: {} mb:{}".format(0, 0, n_samples, 0, L, M, minibatch_size), transform=plt.gcf().transFigure)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.subplots_adjust(left=0.26)
    i = 0
    step = 10
    while i < max_iter:
        net.train(AdamOptimizer(learning_rate), no_epochs=step, minibatch_size=minibatch_size, show_training_info=False, continue_training=True)
        i += step
        testll, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)
        points_y = net.sampleFromPredictive((points_x - X_train_mean) / X_train_std, y_train_std, y_train_mean)

        m, v = net.predict(X_plot)

        # plt.plot(X_train, y_train, 'bo', alpha=0.5)
        line0.set_ydata(points_y)
        line2.set_ydata(m)
        line3.set_ydata(m-2*np.sqrt(v))
        line4.set_ydata(m+2*np.sqrt(v))
        text.set_text("RMSE: {:.4f}, LL: {:.4f} \nSamp: {}, iter: {} \nLayers: {} M: {} mb:{}".format(rmse, testll, n_samples, i, L, M, minibatch_size))

        fig.canvas.draw()

else:
    net.train(AdamOptimizer(learning_rate), no_epochs=max_iter, minibatch_size=minibatch_size, show_training_info=True)

testll, rmse = net.getLogLikelihoodError(X_test, y_test, y_train_std, y_train_mean)
print("Test RMSE: {}".format(rmse))
print("Test ll: {}".format(testll))

m, v = net.predict(X_plot)

#########################################
# Plot results
#########################################

# sns.set()  # seaborn style
if save_to_file and not os.path.exists(folder_name):
    os.makedirs(folder_name)

if samples_from_predictive:
    plt.figure(num=1, figsize=(9, 6))

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
        plt.savefig(f"{folder_name}/alpha{alpha}_{M}_L{L}_S{n_samples}_mb{minibatch_size}_pdf.png")
    else:
        plt.show()

plt.figure(1)
if samples_from_predictive:
    # Sample from predictive dist ex
    plt.plot(points_x, points_y, 'g.', markersize=2)

plt.text(0.01, 0.90, "RMSE: {:.4f}, LL: {:.4f} \nSamp: {}, iter: {} \nLayers: {} M: {} mb:{}".format(rmse, testll, n_samples, max_iter, L, M, minibatch_size), transform=plt.gcf().transFigure)
plt.plot(X_train, y_train, 'bo', alpha=0.5)
plt.plot(X_plot, m, 'm-')
plt.plot(X_plot, m-2*np.sqrt(v), 'm--')
plt.plot(X_plot, m+2*np.sqrt(v), 'm--')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.subplots_adjust(left=0.26)
if save_to_file:
    print(f"Figures are in folder {folder_name}")
    plt.savefig(f"{folder_name}/alpha{alpha}_M{M}_L{L}_S{n_samples}_mb{minibatch_size}.png")
else:
    plt.show()
