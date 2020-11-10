# License: BSD
# Author: Sasank Chilamkurthy
#来自于https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

"""
我们将使用 torchvision 和 torch.utils.data 包来加载数据。

今天，我们要解决的问题是训练一个模型来对蚂蚁和蜜蜂进行分类。
我们蚂蚁和蜜蜂分别准备了大约120个训练图像，并且每类还有75个验证图像。
通常，如果从头开始训练，这是一个非常小的数据集。
由于我们正在使用迁移学习，我们应该能够合理地进行泛化。

该数据集是imagenet的一个很小的子集。
"""

# Data augmentation and normalization for training
# Just normalization for validation

#Compose函数将图片的几种变换组合起来
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), #缩放图片，保持长宽比不变，最短边的长为256像素,
        transforms.CenterCrop(224),#从中间切出 224*224的图片
        transforms.ToTensor(),#将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
"""
ImageFolder是一个通用的数据加载器，数据集中的数据以以下方式组织：
既其默认你的数据集已经自觉按照要分配的类型分成了不同的文件夹，一种类型的文件夹下面只存放一种类型的图片

os.path.join(data_dir, x) 是图片存储的路径
data_transforms[x] 一个函数，原始图片作为输入，返回一个转换后的图片。
x 是 train 或者 val
"""
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

print(type(image_datasets))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

"""
classes 是ImageFolder类的成员：self.classes - 用一个list保存 类名

还有如下成员：
self.class_to_idx - 类名对应的 索引
self.imgs - 保存(img-path, class) tuple的list
"""
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
"""
让我们通过可视化一些训练图像，来理解什么是数据增强。
"""
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

"""
现在, 让我们编写一个通用函数来训练一个模型。这里, 我们将会举例说明:

调整学习率
保存最好的模型
"""
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


"""
模型预测的可视化
用于显示少量预测图像的通用函数
"""
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


"""
微调卷积神经网络
加载预训练模型并重置最后的全连接层。
"""
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device) #Moves and/or casts the parameters and buffers. to device

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

"""
训练与评价
在CPU上训练需要大约15-25分钟。但是在GPU上，它只需不到一分钟。
"""
#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                       num_epochs=25)

"""
将卷积神经网络为固定特征提取器
在这里，我们需要冻结除最后一层之外的所有网络。我们需要设置requires_grad == False来冻结参数，以便在backward()中不会计算梯度。
"""
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False #冻结除最后一层之外的所有网络

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features #全连接层输入的大小
model_conv.fc = nn.Linear(num_ftrs, 2) #重置模型的全连接层

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

"""
训练与评价
在CPU上，与前一个场景相比，大概只花费一半的时间。这在预料之中，因为不需要为绝大多数网络计算梯度。当然，我们还是需要计算前向传播。
"""
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
