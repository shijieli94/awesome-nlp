import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# initialize parameters(w,b)
def initialize_parameters(layer_dims):
    np.random.seed(3)
    L = len(layer_dims)  # the number of layers in the network
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_backward(Z):
    dA = np.int64(Z > 0)
    return dA


# implement the activation function(ReLU and sigmoid)
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def forward_propagation(X, parameters):
    L = len(parameters) // 2  # number of layer
    A = X
    caches = [(None, None, None, X)]  # 第0层(None,None,None,A0) w,b,z用none填充,下标与层数一致，用于存储每一层的，w,b,z,A
    # calculate from 1 to L-1 layer
    for l in range(1, L):
        A_pre = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        z = np.dot(W, A_pre) + b  # 计算z = wx + b
        A = relu(z)  # relu activation function
        caches.append((W, b, z, A))
    # calculate Lth layer
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    zL = np.dot(WL, A) + bL
    AL = sigmoid(zL)
    caches.append((WL, bL, zL, AL))
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = 1.0 / m * np.nansum(np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    cost = np.squeeze(cost)
    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cross_entropy_cost = compute_cost(AL, Y)  # This gives you the cross-entropy part of the cost
    L = len(parameters) // 2
    L2_regularization_cost = 0
    for l in range(0, L):
        L2_regularization_cost += (1.0 / m) * (lambd / 2.0) * (np.sum(np.square(parameters["W" + str(l + 1)])))
    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def backward_propagation_with_regularization(AL, Y, caches, lambd):
    m = Y.shape[1]
    L = len(caches) - 1
    # calculate the Lth layer gradients
    prev_AL = caches[L - 1][3]
    dzL = 1.0 / m * (AL - Y)
    dWL = np.dot(dzL, prev_AL.T) + lambd / m * caches[L][0]
    dbL = np.sum(dzL, axis=1, keepdims=True)
    gradients = {"dW" + str(L): dWL, "db" + str(L): dbL}
    # calculate from L-1 to 1 layer gradients
    for l in reversed(range(1, L)):  # L-1,L-3,....,1
        post_W = caches[l + 1][0]  # 要用后一层的W
        dz = dzL  # 用后一层的dz
        dal = np.dot(post_W.T, dz)
        z = caches[l][2]  # 当前层的z
        dzl = np.multiply(dal, relu_backward(z))  # 可以直接用dzl = np.multiply(dal, np.int64(Al > 0))来实现
        prev_A = caches[l - 1][3]  # 前一层的A
        dWl = np.dot(dzl, prev_A.T) + lambd / m * caches[l][0]
        dbl = np.sum(dzl, axis=1, keepdims=True)

        gradients["dW" + str(l)] = dWl
        gradients["db" + str(l)] = dbl
        dzL = dzl  # 更新dz
    return gradients


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, lambd):
    costs = []
    # initialize parameters
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        # foward propagation
        AL, caches = forward_propagation(X, parameters)
        # calculate the cost
        cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
        # backward propagation
        grads = backward_propagation_with_regularization(AL, Y, caches, lambd)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
    print("length of cost")
    print(len(costs))
    plt.clf()
    plt.plot(costs)  # o-:圆形
    plt.xlabel("iterations(thousand)")  # 横坐标名字
    plt.ylabel("cost")  # 纵坐标名字
    plt.show()
    return parameters


# predict function
def predict(X_test, y_test, parameters):
    m = y_test.shape[1]
    Y_prediction = np.zeros((1, m))
    prob, caches = forward_propagation(X_test, parameters)
    for i in range(prob.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if prob[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    accuracy = 1 - np.mean(np.abs(Y_prediction - y_test))
    return accuracy


# DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.001, num_iterations=20000, lambd=0.0):
    parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, lambd)
    accuracy = predict(X_test, y_test, parameters)
    return accuracy


if __name__ == "__main__":
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, random_state=28)
    X_train = X_train.T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T
    accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1], lambd=0.7)
    print(accuracy)
