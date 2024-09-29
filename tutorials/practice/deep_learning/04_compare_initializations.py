# 对比几种初始化方法
import matplotlib.pyplot as plt
import numpy as np

ACTIVATION = "relu"  # relu, tanh


# 初始化为0
def initialize_parameters_zeros():
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# 随机初始化
def initialize_parameters_random():
    np.random.seed(3)  # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# xavier initialization
def initialize_parameters_xavier():
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            1 / layers_dims[l - 1]
        )
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# He initialization
def initialize_parameters_he():
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2 / layers_dims[l - 1]
        )
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def activate(Z):
    if ACTIVATION == "tanh":
        A = np.tanh(Z)
    elif ACTIVATION == "relu":
        A = np.maximum(0, Z)
    else:
        raise ValueError("Invalid Activation type " + ACTIVATION)
    return A


def forward_propagation():
    A = data
    fig = plt.figure()
    for l in range(1, num_layers):
        A_pre = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        z = np.dot(W, A_pre) + b  # 计算z = wx + b
        A = activate(z)
        plt.subplot(2, 3, l)
        plt.hist(A.flatten(), facecolor="g")
        plt.xlim([-1, 1])
        plt.yticks([])
    fig.suptitle(initialization)
    plt.show()


if __name__ == "__main__":
    data = np.random.randn(1000, 100000)
    layers_dims = [1000, 800, 500, 300, 200, 100, 10]
    num_layers = len(layers_dims)

    for initialization in ["zeros", "random", "xavier", "he"]:
        # Initialize parameters dictionary.
        if initialization == "zeros":
            parameters = initialize_parameters_zeros()
        elif initialization == "random":
            parameters = initialize_parameters_random()
        elif initialization == "xavier":
            parameters = initialize_parameters_xavier()
        elif initialization == "he":
            parameters = initialize_parameters_he()

        forward_propagation()
