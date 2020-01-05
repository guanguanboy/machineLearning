import torch
from torch import nn
print(torch.__version__)

def corr2d(X, K):
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()

    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
res = corr2d(X, K)
print(res)

#二维卷积层的实现
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

#图像中物体边缘检测
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(Y)

#通过数据学习核数组
conv2d = Conv2D(kernel_size=(1, 2)) #构造一个核数据形状是（1， 2）的二维卷积层
step = 20
lr = 0.01

for i in range(step):
    Y_hat = conv2d(X)
    loss = ((Y_hat - Y) ** 2).sum()
    loss.backward()

    #梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    #梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1) % 5 == 0:
        print('Step %d, loss %.3f' % (i+1, loss.item()))


print("weight: ", conv2d.weight.data)
print("bais: ", conv2d.bias.data)