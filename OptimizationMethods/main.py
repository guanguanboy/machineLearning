import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from  testCases import *
def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2 #强制向下取整，这里L计算的是层数：number of layers in neural networks

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

#测试update_parameters_with_gd函数
parameters, grads, learning_rate = update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1=" + str(parameters["W1"]))
print("b1=" + str(parameters["b1"]))
print("W2=" + str(parameters["W2"]))
print("b2=" + str(parameters["b2"]))

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1] #number of training examples
    mini_batches = []

    #Step 1: Shuffle(X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    #Step 2: Partition(shuffled_X, shuffled_Y). Minus the end case
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    #Handling the end case(last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k+1)* mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

#测试random_mini_batches函数
X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3nd mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3nd mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["dW" + str(l + 1)].shape[0], parameters["dW" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["db" + str(l + 1)].shape[0], parameters["db" + str(l + 1)].shape[1]))

    return v

#测试initialize_velocity
parameters = initialize_velocity_test_case()
v = initialize_velocity(parameters)

print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        parameters["dW" + str(l + 1)] = parameters["dW" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["db" + str(l + 1)] = parameters["db" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v

#test update_parameters_with_momentum
parameters, grads, v = update_parameters_with_momentum_test_case()
parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print("v[\"dW1\"] = " + str(v["dW1"]) )
print("v[\"db1\"] = " + str(v["db1"]) )
print("v[\"dW2\"] = " + str(v["dW2"]) )
print("v[\"db2\"] = " + str(v["db2"]) )

