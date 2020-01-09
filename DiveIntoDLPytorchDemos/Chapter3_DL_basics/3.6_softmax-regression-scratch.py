import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") #为了导入上层的d2lzh_pytorch
import d2lzh_pytorch as d2l
print(torch.__version__)
print(torchvision.__version__)

#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.tensor(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  #dim==0表示按列相加，keepdim=True表示在结果中保留两个维度，
print(X.sum(dim=1, keepdim=True))  #dim==1表示每行中的数字相加

#实现softmax运算
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1,keepdim=True))

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))

#定义损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

print(accuracy(y_hat, y))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

print(evaluate_accuracy(test_iter, net))

#训练模型
num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

#预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])