import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision import transforms
import os
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#热狗识别

#1，获取数据集
data_dir = '/liguanlin/projects/tmp/pycharm_project_983/Datasets'
print(data_dir)

os.listdir(os.path.join(data_dir, "hotdog"))
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

#下⾯画出前8 张正例图像和最后8 张负例图像。可以看到，它们的⼤小和⾼宽⽐各不相同。
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]

d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
#plt.show()

"""
在训练时，我们先从图像中裁剪出随机⼤小和随机⾼宽⽐的⼀块随机区域，然后将该区域缩放为
⾼和宽均为224 像素的输⼊。测试时，我们将图像的⾼和宽均缩放为256 像素，然后从中裁剪出
⾼和宽均为224 像素的中⼼区域作为输⼊。此外，我们对RGB（红、绿、蓝）三个颜⾊通道的数值
做标准化：每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。
"""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

#定义和初始化模型
pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

#获取新建模型输出层的参数
output_params = list(map(id, pretrained_net.fc.parameters()))

feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01

optimizer = optim.SGD([{'params' : feature_params},
                       {'params' : pretrained_net.fc.parameters(),
                        'lr' : lr * 10}],
                      lr=lr, weight_decay=0.0001)


#我们先定义⼀个使⽤微调的训练函数train_fine_tuning 以便多次调⽤
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'),
                                        transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'),
                                       transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

#我们将Trainer 实例中的学习率设的小⼀点，例如0.01，以便微调预训练得到的模型参数。根
#据前⾯的设置，我们将以10 倍的学习率从头训练⽬标模型的输出层参数。
train_fine_tuning(pretrained_net, optimizer)

#作为对⽐，我们定义⼀个相同的模型，但将它所有的模型参数都初始化为随机值。由于整个模型
#都需要从头训练，我们可以使⽤较⼤的学习率。
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)

"""
以上训练的结果分别如下：
training on  cuda
epoch 1, loss 4.1677, train acc 0.707, test acc 0.896, time 63.3 sec
epoch 2, loss 0.2149, train acc 0.889, test acc 0.931, time 16.8 sec
epoch 3, loss 0.0662, train acc 0.934, test acc 0.920, time 17.0 sec
epoch 4, loss 0.1405, train acc 0.872, test acc 0.950, time 16.8 sec
epoch 5, loss 0.0455, train acc 0.931, test acc 0.954, time 17.0 sec
training on  cuda
epoch 1, loss 2.7557, train acc 0.671, test acc 0.792, time 17.0 sec
epoch 2, loss 0.2314, train acc 0.807, test acc 0.833, time 16.9 sec
epoch 3, loss 0.1247, train acc 0.828, test acc 0.820, time 16.6 sec
epoch 4, loss 0.0899, train acc 0.840, test acc 0.838, time 16.9 sec
epoch 5, loss 0.0813, train acc 0.814, test acc 0.846, time 17.0 sec

从结果可以看到，微调的模型因为参数初始值更好，往往再相同迭代周期下取得更高的精度
"""

"""
练习：
不断增⼤finetune_net 的学习率。精度会有什么变化？
lr=0.1时
epoch 1, loss 33.9988, train acc 0.492, test acc 0.500, time 17.0 sec
epoch 2, loss 3.4263, train acc 0.529, test acc 0.500, time 16.7 sec
epoch 3, loss 0.8912, train acc 0.511, test acc 0.685, time 16.6 sec
epoch 4, loss 0.2776, train acc 0.642, test acc 0.500, time 16.8 sec
epoch 5, loss 0.2164, train acc 0.580, test acc 0.611, time 16.8 sec

从结果看，学习率从0.01变成0.1之后，精度下降明显

"""