import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn as nn

data_dir = 'Datasets/datasets_detcting'
#加载数据这部分需要重新编写代码，使其可以加载我们自己的图片
#Compose函数将图片的几种变换组合起来
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.Grayscale(),  # 将图片变成灰度图片
        transforms.ToTensor(),#将图片转换为Tensor,归一化至[0,1]
        #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        transforms.Normalize([0.564, 0.564, 0.564], [0.064, 0.064, 0.064])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), #缩放图片，保持长宽比不变，最短边的长为256像素,
        transforms.CenterCrop(224),#从中间切出 224*224的图片
        #transforms.Grayscale(),#将图片变成灰度图片
        transforms.ToTensor(),#将图片转换为Tensor,归一化至[0,1]

        #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        transforms.Normalize([0.564, 0.564, 0.564],[0.079, 0.079, 0.079])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}


# 下载已经具备最优参数的resnet模型
model = models.resnet152(pretrained=True)
print(model)

# 冻结参数
for parma in model.parameters():
    parma.requires_grad = False

#改造最后的全连接层，注意，使用不同的模型进行迁移学习时，这一步是最大的不同
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 1000),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.5),
                         nn.Linear(1000, 1000),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.5),
                         nn.Linear(1000, 6),
                         nn.LogSoftmax(dim=1))


print(model)
#打印待训练的层
params_to_update = model.parameters()
print('Params to learn:')
param_to_update = []

for name, param in model.named_parameters():
    if param.requires_grad == True:
        param_to_update.append(param)
        print('\t', name)

# 判断计算机的GPUs是否可用
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()

# 定义代价函数和优化函数
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=0.00001) #注意这里的第一个参数需要是可优化参数的集合


# 模型训练和参数优化
epoch_n =100
time_open = time.time()
loss_set = []
Acc_set = []
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch + 1, epoch_n))
    print("-" * 10)

    for phase in ["train", "val"]:
        if phase == "train":
            print("Training...")
            # 设置为True，会进行Dropout并使用batch mean和batch var
            model.train(True)
        else:
            print("Validing...")
            # 设置为False，不会进行Dropout并使用running mean和running var
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        # enuerate(),返回的是索引和元素值，数字1表明设置start=1，即索引值从1开始
        for batch, data in enumerate(dataloaders[phase], 1):
            # X: 图片，16*3*224*224; y: 标签，16
            X, y = data

            # 修改处
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            # y_pred: 预测概率矩阵，16*2
            y_pred = model(X)

            # pred，概率较大值对应的索引值，可看做预测结果
            _, pred = torch.max(y_pred.data, 1)
            #print('ypred:')
            #print(pred)
           # print(pred.type)
            # 梯度归零
            optimizer.zero_grad()

            # 计算损失
            loss = loss_f(y_pred, y)
            #print('y:')
            #print(y)
           # print(type(y))
            #print(y.shape)
           # res = y[0].numel()
            #print('res:')
            #print(res)
            # 若是在进行模型训练，则需要进行后向传播及梯度更新
            if phase == "train":
                loss.backward()
                optimizer.step()

            # 计算损失和
            running_loss += float(loss)

            # 统计预测正确的图片数
            running_corrects += torch.sum(pred == y.data)
            #print(pred)
            # 共20000张测试图片，1250个batch，在使用500个及1000个batch对模型进行训练之后，输出训练结果
            if batch % 500 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4F}%".format(batch, running_loss / batch,
                                                                              100 * running_corrects / (16 * batch)))
          #  print(y)
          #  confusion = confusion_matrix(y.data, pred)

        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        if phase == "train":
            loss_set.append(epoch_loss)
            Acc_set.append(epoch_acc)
        # 输出最终的结果
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))



