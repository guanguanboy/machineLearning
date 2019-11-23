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

def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    x = x / x_norm

    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]
])

print("normalizeRows(x) = " + str(normalizeRows(x)))

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims=True)
    s = x_exp / x_sum

    print("x_exp.shape =" + str(x_exp.shape))
    print("x_sum.shape =" + str(x_sum.shape))
    print("s.shape = " + str(s.shape))
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))

def L1(y_hat, y):
    loss = np.sum(abs(y_hat - y))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(y_hat, y):
    loss = np.sum(np.dot(abs(y - y_hat), abs(y - y_hat)))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))

