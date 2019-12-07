import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from main import nn_model, predict

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs":blobs,
            "gaussian_quantiles":gaussian_quantiles}
dataset_name = "gaussian_quantiles"

X, Y = datasets[dataset_name]
print(X.shape) #(200, 2)
print(Y.shape) #(200,)

X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset_name == "blobs":
    Y = Y % 2

plt.scatter(X[0, :], X[1, :], c = Y[0,:], s = 40, cmap=plt.cm.Spectral)


parameters = nn_model(X, Y, 5, num_iterations=5000)
plot_decision_boundary(lambda x : predict(parameters, x.T), X, Y)
predictions = predict(parameters, X)
accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
print("Accuracy for {} hidden units: {} %".format(5, accuracy))

plt.show()
