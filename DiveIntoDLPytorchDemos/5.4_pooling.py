import torch
from torch import nn

print(torch.__version__)

#二维最大池化层和平均池化层
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j + p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))

print(pool2d(X, (2, 2), 'avg'))

#填充和步幅
X = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
pool2d = nn.MaxPool2d(3)
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
res = pool2d(X)
print(res)

pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
res = pool2d(X)

print(res)

#多通道
"""
在处理多通道的输入数据时， 池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按
通道相加。这意味着池化层的输出通道数与输入通道数相等。
"""
X = torch.cat((X, X+1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))



