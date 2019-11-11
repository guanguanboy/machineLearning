import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *


#%matplotlib inline jupyter可以用这句话来简化绘图流程，这样就不需要调用plt.show()
#https://www.cnblogs.com/pacino12134/p/9776882.html
plt.rcParams['figure.figsize'] = (7.0, 4.0) #   # 图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray' # 使用灰度输出而不是彩色输出

train_X, train_Y, test_X, test_Y = load_2D_dataset()

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

plt.scatter(train_X[0, :], train_X[1, :], c=train_Y[0,:], s=40, cmap=plt.cm.Spectral)
plt.show()

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y) #This gives you the cross-entropy part of the cost

    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lamdb):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dz3 = A3 - Y

    dW3 = 1./m * np.dot(dz3, A2.T) + lamdb * W3 / m

    db3 = 1./m * np.sum(dz3, axis=1, keepdims= True)

    dA2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(dA2, np.int64(A2 > 0))

    dW2 = 1./m * np.dot(dz2, A1.T) + lamdb * W2 / m
    db2 = 1./m * np.sum(dz2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dz1, X.T) + lamdb * W1 / m
    db1 = 1./m * np.sum(dz1, axis=1, keepdims=True)

    graients = {"dz3": dz3, "dW3":dW3, "db3": db3, "dA2":dA2,
                "dz2": dz2, "dW2":dW2, "db2": db2, "dA1":dA1,
                "dz1": dz1, "dW1":dW1, "db1": db1}

    return graients


#Non-regularized model
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lamdb = 0, keep_prob = 1):
    """
     Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

     Arguments:
     X -- input data, of shape (input size, number of examples)
     Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
     learning_rate -- learning rate of the optimization
     num_iterations -- number of iterations of the optimization loop
     print_cost -- If True, print the cost every 10000 iterations
     lambd -- regularization hyperparameter, scalar
     keep_prob - probability of keeping a neuron active during drop-out, scalar.

     Returns:
     parameters -- parameters learned by the model. They can then be used to predict.
     """
    grads = {}
    costs = [] #to keep track of the cost

    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    parameters = initialize_parameters(layers_dims)

    #loop graident descent
    for i in range(0, num_iterations):
        #forward propagation:
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        #elif keep_prob < 1:
            #a3, cache = forward_propagation_with_dropout()

        #Cost function
        if lamdb == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lamdb)

        #Backward propagation.
        assert(lamdb==0 or keep_prob==1)

        if lamdb==0 and keep_prob==1:
            grads = backward_propagation(X, Y, cache)
        elif lamdb != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lamdb)
        #elif keep_prob < 1:
         #   grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        #Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))

        if print_cost and i % 1000 == 0:
            costs.append(cost)
    #plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#train the model without regularization
parameters = model(train_X, train_Y)
predictions_train = predict(train_X, train_Y, parameters)
predictions_test = predict(test_X, test_Y, parameters)