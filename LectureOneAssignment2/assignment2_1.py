import math
import numpy as np

def basic_sigmoid(x):
    s = 1.0 / (1 + 1.0/ math.exp(x))

    return s

print(basic_sigmoid(3))

x = [1, 2, 3]
#basic_sigmoid(x) #会发生错误，输出TypeError: must be real number, not list，math.exp()只能处理real number

x_np = np.array([1, 2, 3])
print(np.exp(x_np))

def sigmoid(x):
    s = 1.0 / (1 + 1.0 / np.exp(x))

    return s

x = np.array([1, 2, 3])
sigmoid(x)

#sigmoid gradient

def sigmoid_derivative(x):
    s = 1.0 / (1 + 1.0 / np.exp(x))
    ds = s * (1 - s)

    return ds

x = np.array([1, 2, 3])
print("sigmoid_derivative(x) =" + str(sigmoid_derivative(x)))
