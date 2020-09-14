import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cpu')
print(device)

#9.1.1 常用的图像增广方法
d2l.set_figsize() #设置图像大小，默认大小为figsize=(3.5, 2.5)
img = Image.open('img/cat1.jpg')
d2l.plt.imshow(img)
#plt.imshow(img)

plt.show()

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

#大部分的图像增广方法都有一定的随机性。为了方便观察图像增广的效果，接下来我们定义一个辅助函数
#apply.该函数对输入图像img多次运行图像增广方法aug并展示所有的结果。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
    plt.show()

#翻转和裁剪

#左右翻转图像通常不改变物体的类别。它是最早也是最广泛使用的一种图像增广方法。
#下面我们通过transforms模块创建RandomHorizontalFlip实例来实现一半概率的图像左右翻转
#apply(img, torchvision.transforms.RandomHorizontalFlip())

#一半概率的图像的上下翻转
#apply(img, torchvision.transforms.RandomVerticalFlip())

#在下⾯的代码⾥，我们每次随机裁剪出⼀块⾯积为原⾯积10% 到100% 的区域，且该区域的宽
#和⾼之⽐随机取⾃0.5 和2 之间，然后再将该区域的宽和⾼分别缩放到200 像素。如⽆特殊说明，
#本节中a 和b 之间的随机数指的是从区间[a; b] 中均匀采样所得到的连续值。
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
#apply(img, shape_aug)

#变化颜色
#另⼀类增⼴⽅法是变化颜⾊。我们可以从四个⽅⾯改变图像的颜⾊：亮度、对⽐度、饱和度和⾊
#调。在下⾯的例⼦⾥，我们将图像的亮度随机变化为原图亮度的50%（1-0.5）到150%（1+0.5）之间。
#brightness:表示亮度，contrast表示对比度，saturation表示饱和度，hue:表示色调
#apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

#随机变化图像的色调
"""
关于色调的解释
色调指的是一幅画中画面色彩的总体倾向，是大的色彩效果。
色调是颜色的重要特征，它决定了颜色本质的根本特征。
色调是由物体反射光线中以哪种波长占优势来决定的，不同波长产生不同颜色的感觉，
如在大自然中，经常见到这样一种现象：
不同颜色的物体或被笼罩在一片金色的阳光之中，
或被笼罩在一片轻纱薄雾似的、淡蓝色的月色之中；
或被秋天迷人的金黄色所笼罩；或被统一在冬季银白色的世界之中。
这种在不同颜色的物体上，笼罩着某一种色彩，
使不同颜色的物体都带有同一色彩倾向，这样的色彩现象就是色调。
"""
#apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

#随机变化图像的对比度
#apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0))

"""
我们也可以创建ColorJitter 实例并同时设置如何随机变化图像的亮度（brightness）
、对⽐度（contrast）、饱和度（saturation）和⾊调（hue）
"""
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
#apply(img, color_aug)

#实际应⽤中我们会将多个图像增⼴⽅法叠加使⽤。
# 我们可以通过Compose 实例将以上定义的多
#个图像增⼴⽅法叠加起来，再应⽤到每个图像之上。
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,shape_aug
])

#apply(img, augs)

#9.1.2 使用图像增广训练模型
#1，下载数据
all_images = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)
#all_imges的每一个元素都是(image, label)
show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
plt.show()
#2,增广数据
#为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用
#含随机操作的图像增广。在这里我们仅仅使用最简单的随机左右翻转。
#此外，我们使用ToTensor实例将小批量图像转成MXNet需要的格式。
flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

num_workers = 0

def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

#训练模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

#有关图像增广的对比实验
#1，使用了图像增广的结果
#train_with_data_aug(flip_aug, no_aug)

#2，不使用图像增广的结果
train_with_data_aug(no_aug, no_aug)

#可以看到，即使添加了简单的随机翻转也可能对训练产⽣⼀定的影响。
#  图像增⼴通常会使训练准确率变低，但有可能提⾼测试准确率。它可以⽤来应对过拟合。