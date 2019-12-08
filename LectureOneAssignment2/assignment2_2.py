import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#load the data(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Example of a picture
index = 16
#plt.imshow(train_set_x_orig[index])
#plt.show()
print("y=" + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
num_px_height = train_set_x_orig.shape[2]
print("Number of trainning examples: m_train =" + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Width of each image: num_px = " + str(num_px))
print("Height of each image: num_px = " + str(num_px_height))
print("Each image is of size: (" + str(num_px) + ")" + str(num_px_height) + ", 3")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

#Reshape the trainning and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

#standardize our dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#4.1 helper functions
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s

print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

#4.2 initializing parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b

dim = 10
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))

#4.3 forward and backward propagation
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b) #shape==(1, number of examples)
    cost = -(np.sum(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)))/m

    #backward propagation
    dw = (np.dot(X, (A-Y).T))/m
    db = (np.sum(A-Y))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw":dw,
             "db":db}

    return grads, cost

w,b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)

print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw":dw,
             "db":db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print("w =" + str(params["w"]))
print("b =" + str(params["b"]))
print("dw =" + str(grads["dw"]))
print("db =" + str(grads["db"]))

def predict(w, b, X):
    m = X.shape[1]

    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[:, i] >=0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

print("predictions = " + str(predict(w, b, X)))

#5,merge all functions into a model
def model(X_train, Y_train, X_test, Y_test, num_iteerations = 2000, learning_rate=0.5, print_cost = False):

    w,b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iteerations, learning_rate, print_cost=True)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train))* 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iteerations
    }

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteerations=2000, learning_rate=0.005, print_cost=True)

#plot learning curve
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


#choice of learning rate
learning_rates = [0.01, 0.001, 0.0001, 0.02, 0.00001]
models = {}

for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x,test_set_y, num_iteerations=1500, learning_rate=i, print_cost=False)
    print('\n' + "---------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()