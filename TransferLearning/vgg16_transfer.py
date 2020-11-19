import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import confusion_matrix

"""
data_dir = './data/DogsVSCats'
# 定义要对数据进行的处理
data_transform = {x: transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                  for x in ["train", "valid"]}
# 数据载入
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "valid"]}
# 数据装载
dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                             batch_size=16,
                                             shuffle=True)
              for x in ["train", "valid"]}

X_example, y_example = next(iter(dataloader["train"]))
# print(u'X_example个数{}'.format(len(X_example)))
# print(u'y_example个数{}'.format(len(y_example)))
# print(X_example.shape)
# print(y_example.shape)

# 验证独热编码的对应关系
index_classes = image_datasets["train"].class_to_idx
# print(index_classes)
# 使用example_classes存放原始标签的结果
example_classes = image_datasets["train"].classes
# print(example_classes)

# 图片预览
img = torchvision.utils.make_grid(X_example)
# print(img.shape)
img = img.numpy().transpose([1, 2, 0])

for i in range(len(y_example)):
    index = y_example[i]
    print(example_classes[index], end='   ')
    if (i + 1) % 8 == 0:
        print()

# print(img.max())
# print(img.min())
# print(img.shape)

std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean

# print(img.max())
# print(img.min())
# print(img.shape)

plt.imshow(img)
plt.show()
"""
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

print(image_datasets['train'][0][0].size)

image, label = next(iter(image_datasets['train']))

print('image_shape:')
print(image.shape)

#print(type(image_datasets))
#print(image_datasets)


class_names = image_datasets['train'].classes

print(class_names)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# 下载已经具备最优参数的VGG16模型
model = models.vgg16(pretrained=True)
print(model)
# 查看迁移模型细节
# print("迁移VGG16:\n", model)

# 对迁移模型进行调整
for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 6))

# 查看调整后的迁移模型
# print("调整后VGG16:\n", model)
# 判断计算机的GPUs是否可用
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

# 定义代价函数和优化函数
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)

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


tt = np.linspace(1, epoch_n, num=epoch_n)
plt.figure()
plt.plot(tt, loss_set, '-')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(tt, Acc_set, '-')
plt.xlabel('Echo')
plt.ylabel('Acc')
plt.show()

origin_labels = []
pred_labels=[]

#使用训练好的模型对所有数据重新预测一遍
for batch, data in enumerate(dataloaders['train'], 1):
    # X: 图片，16*3*224*224; y: 标签，16
    X, y = data

    # 修改处
    if Use_gpu:
        X, y = Variable(X.cuda()), Variable(y.cuda())
    else:
        X, y = Variable(X), Variable(y)

    # y_pred: 预测概率矩阵，16*2
    y_pred = model(X)
    _, pred = torch.max(y_pred.data, 1)

    print('y:')
    print(y)
    print(type(y))
    print(y.shape)
    res = y[0].numel()
    print('res:')
    print(res)

    origin_labels_list = y.cpu().numpy().tolist()

    for i in range(len(origin_labels_list)):
        origin_labels.append(origin_labels_list[i])

    #origin_labels.append(y[1].numel())
    #origin_labels.append(y[2].numel())
    #origin_labels.append(y[3].numel())

    pred_labels_list = pred.cpu().numpy().tolist()

    for i in range(len(pred_labels_list)):
        pred_labels.append(pred_labels_list[i])

    #pred_labels.append(pred[1].numel())
    #pred_labels.append(pred[2].numel())
    #pred_labels.append(pred[3].numel())

#confusion = confusion_matrix(origin_labels, pred_labels)
#print(confusion)

labels = ['jiaza', 'liefeng', 'qikong', 'weihanjie', 'weironghe', 'zhengchang']

tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#ndarray_origin = np.array(origin_labels)
#y_true = (ndarray_origin.flatten()).tolist()

#y_pred = ((np.array(pred_labels)).flatten()).tolist()
y_true = origin_labels
y_pred = pred_labels

print(y_true)
print(len(y_true))
print(y_pred)
print(len(y_pred))

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('./data/confusion_matrix.png', format='png')
plt.show()

# 输出模型训练、参数优化用时
time_end = time.time() - time_open
print(time_end)