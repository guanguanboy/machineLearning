import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") #为了导入上层目录的d2lzh_pytorch

import d2lzh_pytorch as d2l

print(torch.__version__)
print(torchvision.__version__)

#获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_test))
print(len(mnist_train), len(mnist_test))

#获取第一个样本的图像和标签
feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    figs = plt.subplots(1, len(images), figsize=(12, 12), sharey=True)
    i = 1
    for img, lbl in zip(images, labels):
        plt.subplot(1, 10, i)
        #fig = figs[i]
        plt.imshow(img.view((28, 28)).numpy())
        #plt.axes.set_title(lbl)
        #plt.axes.get_xaxis().set_visible(False)
        #plt.axes.get_yaxis().set_visible(False)
        i = i + 1
    plt.show()

X, y = [], []

for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
    show_fashion_mnist(X, get_fashion_mnist_labels(y))

#获取小批量
