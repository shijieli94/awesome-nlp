from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

PARAMETERS = namedtuple("PARAMETERS", ["W", "b"])
CACHES = namedtuple("CACHES", ["W", "b", "Z", "A"])
GRADIENTS = namedtuple("GRADIENTS", ["dW", "db"])


# initialize parameters(w,b)
def initialize_parameters(layer_dims):
    np.random.seed(3)  # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    for l in range(1, len(layer_dims)):
        W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        b = np.zeros((layer_dims[l], 1))
        parameters[f"L{l}"] = PARAMETERS(W=W, b=b)
    return parameters


# implement the ReLU function
def relu(Z):
    A = np.maximum(0, Z)
    return A


# implement the sigmoid function
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


# calculate cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 1.0 / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    cost = np.squeeze(cost)
    return cost


def forward_propagation(X, parameters):
    L = len(parameters)  # number of layer
    A = X
    caches = {}  # 用于存储每一层的，w,b,z,A
    for l in range(1, L):  # calculate from 1 to L-1 layer
        W = parameters[f"L{l}"].W
        b = parameters[f"L{l}"].b
        Z = np.dot(W, A) + b  # 计算 z = wx + b
        caches[f"L{l}"] = CACHES(W=W, b=b, Z=Z, A=A)

        A = relu(Z)  # relu activation function

    # calculate last layer
    WL = parameters[f"L{L}"].W
    bL = parameters[f"L{L}"].b
    ZL = np.dot(WL, A) + bL
    caches[f"L{L}"] = CACHES(W=WL, b=bL, Z=ZL, A=A)

    AL = sigmoid(ZL)

    return AL, caches


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches)
    # calculate the Lth layer gradients
    dzL = 1.0 / m * (AL - Y)  # 这个是对sigmoid输入的导数

    prev_AL = caches[f"L{L}"].A  # Z = WA + b

    dWL = np.dot(dzL, prev_AL.T)
    dbL = np.sum(dzL, axis=1, keepdims=True)

    gradients = {f"L{L}": GRADIENTS(dW=dWL, db=dbL)}

    # calculate from L-1 to 1 layer gradients
    dz = dzL  # 用后一层的dz
    for l in reversed(range(1, L)):
        post_W = caches[f"L{l + 1}"].W  # 首先求得above一层对当前层的输出的导数
        dA = np.dot(post_W.T, dz)

        z = caches[f"L{l}"].Z  # 由当前层的z决定有梯度的位置
        dz = np.multiply(dA, np.int64(z > 0))

        A = caches[f"L{l}"].A  # 当前层的输入, Z = WA + b
        dW = np.dot(dz, A.T)
        db = np.sum(dz, axis=1, keepdims=True)

        gradients[f"L{l}"] = GRADIENTS(dW=dW, db=db)

    return gradients


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)
    for l in range(1, L + 1):
        W = parameters[f"L{l}"].W - learning_rate * grads[f"L{l}"].dW
        b = parameters[f"L{l}"].b - learning_rate * grads[f"L{l}"].db
        parameters[f"L{l}"] = PARAMETERS(W=W, b=b)
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
    costs = []
    # initialize parameters
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        # forward propagation
        AL, caches = forward_propagation(X, parameters)
        # calculate the cost
        cost = compute_cost(AL, Y)
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
        # backward propagation
        grads = backward_propagation(AL, Y, caches)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
    plt.clf()
    plt.plot(costs)
    plt.xlabel("iterations(thousand)")  # 横坐标名字
    plt.ylabel("cost")  # 纵坐标名字
    plt.show()
    return parameters


# predict function
def predict(X_test, y_test, parameters):
    m = y_test.shape[1]
    pos_prediction = np.ones((1, m))
    neg_prediction = np.zeros((1, m))
    prob, caches = forward_propagation(X_test, parameters)

    Y_prediction = np.where(prob > 0.5, pos_prediction, neg_prediction)

    accuracy = 1 - np.mean(np.abs(Y_prediction - y_test))
    return accuracy


# DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.01, num_iterations=15000):
    parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
    accuracy = predict(X_test, y_test, parameters)
    return accuracy


if __name__ == "__main__":
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=28)
    X_train = X_train.T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T
    accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 20, 10, 5, 1])
    print(accuracy)
