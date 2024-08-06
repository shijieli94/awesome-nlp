# implement the batch normalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# initialize parameters(w,b)
def initialize_parameters(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)  # the number of layers in the network
    parameters = {}
    # initialize the exponential weight average
    bn_param = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        parameters["gamma" + str(l)] = np.ones((layer_dims[l], 1))
        parameters["beta" + str(l)] = np.zeros((layer_dims[l], 1))
        bn_param["moving_mean" + str(l)] = np.zeros((layer_dims[l], 1))
        bn_param["moving_var" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters, bn_param


def relu_forward(Z):
    A = np.maximum(0, Z)
    return A


# implement the activation function(ReLU and sigmoid)
def sigmoid_forward(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def linear_forward(X, W, b):
    z = np.dot(W, X) + b
    return z


def batchnorm_forward(z, gamma, beta, epsilon=1e-12):
    mu = np.mean(z, axis=1, keepdims=True)  # axis=1按行求均值
    var = np.var(z, axis=1, keepdims=True)
    sqrt_var = np.sqrt(var + epsilon)
    z_norm = (z - mu) / sqrt_var
    z_out = np.multiply(gamma, z_norm) + beta  # 对应元素点乘
    return z_out, mu, var, z_norm, sqrt_var


def forward_propagation(X, parameters, bn_param, decay=0.9):
    L = len(parameters) // 4  # number of layer
    A = X
    caches = []
    # calculate from 1 to L-1 layer
    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        gamma = parameters["gamma" + str(l)]
        beta = parameters["beta" + str(l)]
        z = linear_forward(A, W, b)
        z_out, mu, var, z_norm, sqrt_var = batchnorm_forward(z, gamma, beta)  # batch normalization
        caches.append((A, W, b, gamma, sqrt_var, z_out, z_norm))  # 以激活单元为分界线，把做激活前的变量放在一起，激活后可以认为是下一层的x了
        A = relu_forward(z_out)  # relu activation function
        # exponential weight average for test
        bn_param["moving_mean" + str(l)] = decay * bn_param["moving_mean" + str(l)] + (1 - decay) * mu
        bn_param["moving_var" + str(l)] = decay * bn_param["moving_var" + str(l)] + (1 - decay) * var
    # calculate Lth layer(last layer)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    zL = linear_forward(A, WL, bL)
    caches.append((A, WL, bL, None, None, None, None))
    AL = sigmoid_forward(zL)
    return AL, caches, bn_param


# calculate cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 1.0 / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    cost = np.squeeze(cost)
    return cost


# derivation of relu
def relu_backward(dA, Z):
    dout = np.multiply(dA, np.int64(Z > 0))
    return dout


def batchnorm_backward(dout, cache):
    _, _, _, gamma, sqrt_var, _, Z_norm = cache
    m = dout.shape[1]
    dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)  # *作用于矩阵时为点乘
    dbeta = np.sum(dout, axis=1, keepdims=True)
    dy = (
        1.0
        / m
        * gamma
        * sqrt_var
        * (m * dout - np.sum(dout, axis=1, keepdims=True) - Z_norm * np.sum(dout * Z_norm, axis=1, keepdims=True))
    )
    return dgamma, dbeta, dy


def linear_backward(dZ, cache):
    A, W, _, _, _, _, _ = cache
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    da = np.dot(W.T, dZ)
    return da, dW, db


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches) - 1
    # calculate the Lth layer gradients
    dz = 1.0 / m * (AL - Y)
    da, dWL, dbL = linear_backward(dz, caches[L])
    gradients = {"dW" + str(L + 1): dWL, "db" + str(L + 1): dbL}
    # calculate from L-1 to 1 layer gradients
    for l in reversed(range(0, L)):  # L-1,L-3,....,1
        # relu_backward->batchnorm_backward->linear backward
        A, w, b, gamma, sqrt_var, z_out, z_norm = caches[l]
        # relu backward
        dout = relu_backward(da, z_out)
        # batch normalization
        dgamma, dbeta, dz = batchnorm_backward(dout, caches[l])
        # linear backward
        da, dW, db = linear_backward(dz, caches[l])
        # gradient
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db
        gradients["dgamma" + str(l + 1)] = dgamma
        gradients["dbeta" + str(l + 1)] = dbeta
    return gradients


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 4
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        if l < L - 1:
            parameters["gamma" + str(l + 1)] = (
                parameters["gamma" + str(l + 1)] - learning_rate * grads["dgamma" + str(l + 1)]
            )
            parameters["beta" + str(l + 1)] = (
                parameters["beta" + str(l + 1)] - learning_rate * grads["dbeta" + str(l + 1)]
            )
    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=1):
    np.random.seed(seed)
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = (
        m // mini_batch_size
    )  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, mini_batch_size=64):
    costs = []
    # initialize parameters
    parameters, bn_param = initialize_parameters(layer_dims)
    seed = 0
    for i in range(0, num_iterations):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # forward propagation
            AL, caches, bn_param = forward_propagation(minibatch_X, parameters, bn_param)
            # calculate the cost
            cost = compute_cost(AL, minibatch_Y)
            # backward propagation
            grads = backward_propagation(AL, minibatch_Y, caches)
            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate)
        if i % 200 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
    print("length of cost")
    print(len(costs))
    plt.clf()
    plt.plot(costs)  # o-:圆形
    plt.xlabel("iterations(thousand)")  # 横坐标名字
    plt.ylabel("cost")  # 纵坐标名字
    plt.show()
    return parameters, bn_param


# fp for test
def forward_propagation_for_test(X, parameters, bn_param, epsilon=1e-12):
    L = len(parameters) // 4  # number of layer
    A = X
    # calculate from 1 to L-1 layer
    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        gamma = parameters["gamma" + str(l)]
        beta = parameters["beta" + str(l)]
        z = linear_forward(A, W, b)
        # batch normalization
        # exponential weight average
        moving_mean = bn_param["moving_mean" + str(l)]
        moving_var = bn_param["moving_var" + str(l)]
        sqrt_var = np.sqrt(moving_var + epsilon)
        z_norm = (z - moving_mean) / sqrt_var
        z_out = np.multiply(gamma, z_norm) + beta  # 对应元素点乘
        # relu forward
        A = relu_forward(z_out)  # relu activation function

    # calculate Lth layer(last layer)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    zL = linear_forward(A, WL, bL)
    AL = sigmoid_forward(zL)
    return AL


# predict function
def predict(X_test, y_test, parameters, bn_param):
    m = y_test.shape[1]
    Y_prediction = np.zeros((1, m))
    prob = forward_propagation_for_test(X_test, parameters, bn_param)
    for i in range(prob.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if prob[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    accuracy = 1 - np.mean(np.abs(Y_prediction - y_test))
    return accuracy


# DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.01, num_iterations=10000, mini_batch_size=64):
    parameters, bn_param = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, mini_batch_size)
    train_accuracy = predict(X_train, y_train, parameters, bn_param)
    test_accuracy = predict(X_test, y_test, parameters, bn_param)
    return train_accuracy, test_accuracy


if __name__ == "__main__":
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=28)
    X_train = X_train.T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T

    train_accuracy, test_accuracy = DNN(
        X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], mini_batch_size=256
    )
    print("train accuracy: ", train_accuracy)
    print("test accuracy: ", test_accuracy)
