import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# initialize parameters(w,b)
def initialize_parameters(layer_dims):
    np.random.seed(1)
    L = len(layer_dims)  # the number of layers in the network
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# implement the activation function(ReLU and sigmoid)
def relu(Z):
    A = np.maximum(0, Z)
    return A


# derivation of relu
def relu_backward(Z):
    dA = np.int64(Z > 0)
    return dA


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


def backward_propagation(AL, Y, caches):
    m = Y.shape[1]
    L = len(caches) - 1
    # calculate the Lth layer gradients
    prev_AL = caches[L - 1][3]
    dzL = 1.0 / m * (AL - Y)
    dWL = np.dot(dzL, prev_AL.T)
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
        dWl = np.dot(dzl, prev_A.T)
        dbl = np.sum(dzl, axis=1, keepdims=True)

        gradients["dW" + str(l)] = dWl
        gradients["db" + str(l)] = dbl
        dzL = dzl  # 更新dz
    return gradients


# convert parameter into vector
def dictionary_to_vector(parameters):
    count = 0
    for key in parameters:
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))  # convert matrix into vector
        if count == 0:  # 刚开始时新建一个向量
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)  # 和已有的向量合并成新向量
        count = count + 1

    return theta


# convert gradients into vector
def gradients_to_vector(gradients):
    # 因为gradient的存储顺序是{dWL,dbL,....dW2,db2,dW1,db1}，为了统一采用[dW1,db1,...dWL,dbL]方面后面求欧式距离（对应元素）
    L = len(gradients) // 2
    keys = []
    for l in range(L):
        keys.append("dW" + str(l + 1))
        keys.append("db" + str(l + 1))
    count = 0
    for key in keys:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))  # convert matrix into vector
        if count == 0:  # 刚开始时新建一个向量
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)  # 和已有的向量合并成新向量
        count = count + 1

    return theta


# convert vector into dictionary
def vector_to_dictionary(theta, layer_dims):
    parameters = {}
    L = len(layer_dims)  # the number of layers in the network
    start = 0
    end = 0
    for l in range(1, L):
        end += layer_dims[l] * layer_dims[l - 1]
        parameters["W" + str(l)] = theta[start:end].reshape((layer_dims[l], layer_dims[l - 1]))
        start = end
        end += layer_dims[l] * 1
        parameters["b" + str(l)] = theta[start:end].reshape((layer_dims[l], 1))
        start = end
    return parameters


def gradient_check(parameters, gradients, X, Y, layer_dims, epsilon=1e-7):
    parameters_vector = dictionary_to_vector(parameters)  # parameters_values
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_vector.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute grad approx
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_vector)
        thetaplus[i] = thetaplus[i] + epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaplus, layer_dims))
        J_plus[i] = compute_cost(AL, Y)

        thetaminus = np.copy(parameters_vector)
        thetaminus[i] = thetaminus[i] - epsilon
        AL, _ = forward_propagation(X, vector_to_dictionary(thetaminus, layer_dims))
        J_minus[i] = compute_cost(AL, Y)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m"
        )
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m"
        )

    return difference


if __name__ == "__main__":
    X_data, y_data = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=28)
    X_train = X_train.T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T

    # 根据自己实现的bp计算梯度
    parameters = initialize_parameters([X_train.shape[0], 5, 3, 1])
    AL, caches = forward_propagation(X_train, parameters)
    cost = compute_cost(AL, y_train)
    gradients = backward_propagation(AL, y_train, caches)
    # gradient checking
    difference = gradient_check(parameters, gradients, X_train, y_train, [X_train.shape[0], 5, 3, 1])
