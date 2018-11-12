import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def load_dataset():
    # TODO: Load the data

def gradient_descent(X_train, y_train, X_test, y_test, lamda, eta):
    d = X_train.shape[1]
    w = np.zeros(shape=(d, 1))
    b = 0

    list_loss_train = []
    list_loss_test = []
    list_err_train = []
    list_err_test = []
    max_change = 1
    i = -1
    while max_change > 0.0001: #for i in range(200):
        dw, db = get_gradient(X_train, y_train, w, b, lamda)
        w_new = w - dw * eta
        b_new = b - db * eta
        loss_train = get_loss(X_train, y_train, w_new, b_new, lamda)
        loss_test = get_loss(X_test, y_test, w_new, b_new, lamda)
        list_loss_train.append(loss_train)
        list_loss_test.append(loss_test)
        list_err_train.append(get_error(X_train, y_train, w_new, b_new))
        list_err_test.append(get_error(X_test, y_test, w_new, b_new))

        max_change = np.amax(np.abs(w_new - w))
        w = w_new.copy()
        b = b_new
        i = i+1
        print(i, loss_train, loss_test, db)
    plot_loss(list_loss_train, list_loss_test, "Loss for Gradient Descent")
    plot_error(list_err_train, list_err_test, "Error Rate for Gradient Descent")
    return w, b

    def get_error(X, y, w, b):
    return 1 - np.mean(np.sign(np.dot(X, w) + b) == y)

def plot_error(train, test, title):
    plt.plot(range(len(train)), train, label='Train')
    plt.plot(range(len(test)), test, label='Test')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss(train, test, title):
    plt.plot(range(len(train)), train, label='Train')
    plt.plot(range(len(train)), test, label='Test')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

def get_loss(X, y, w, b, lamda):
    y_exp = np.dot(X, w) + b
    sign = -1 * y * y_exp
    mean = np.mean(np.log(1 + np.exp(sign)))
    return mean + (lamda * (np.linalg.norm(w) ** 2))

def get_gradient(X, y, w, b, lamda):
    n, d = X.shape
    w_result = np.zeros(shape=(1, d))
    b_result = 0
    for i in range(n):
        Xi = X[i, :].reshape((1, d))
        common = ((get_mu_i(Xi, y[i], w, b) - 1) * y[i])[0]
        b_result = b_result + common
        w_result = w_result + (common * Xi)
    b_result = b_result / n
    w_result = (w_result/ n) + 2 * lamda * w.T
    return w_result.T, b_result

def get_mu_i(X, y, w, b):
    mu = np.exp(-1 * y * np.dot(X, w)) + 1
    return (1.0 / mu[0, 0])

def get_mu(X, y, w, b):
    y_exp = np.dot(X, w) + b
    sign = (-1 * y) * y_exp
    mu = 1.0 / (1 + np.exp(sign))
    return mu
