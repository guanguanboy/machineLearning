"""
利用resnet50 提取到的特征，构建一个针对dogs and breeds进行分类的网络
根据main中的分析
这样再训练时，我们只需将这个数组读出来，然后可以直接使用这个数组再输入到linear或者
我们前面讲到的sigmoid层就可以了。
我们在这里在pool层前获取了更多的特征，可以将这些特征使用更高级的分类器，例如SVM，
树形分类器进行分类
"""
import torch,os, torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

features = np.load("features.npy")
print("type", type(features))
print("shape:", features.shape)
print("size", features.size)

"""
运行结果：
type <class 'numpy.ndarray'>
shape: (9199, 2048, 7, 7)
size 923138048

9199 是训练用的样本数
"""

DATA_ROOT = 'E:\datasets\dog-breed-identification'
all_labels_df = pd.read_csv(os.path.join(DATA_ROOT, 'labels.csv'))

breeds = all_labels_df.breed.unique()
breed2idx = dict((breed, idx) for idx, breed in enumerate(breeds))
idx2breed = dict( (idx,breed) for idx, breed in enumerate(breeds))

"""
main中用于训练分类 dogs and breeds 的 最后两层信息
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=120, bias=True)
  
resnet50 原本最后一层的信息
Linear(in_features=2048, out_features=1000, bias=True)
"""


class ResTransferNet(nn.Module):
    def __init__(self):
        super(ResTransferNet, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, len(breeds))

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        return x


res_transfer_net = ResTransferNet()


#设置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params':res_transfer_net.fc.parameters()}
], lr=0.001)

#定义训练函数
def train(model,device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat,y)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))

#定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            x,y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat,y).item() #sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] #get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_dataset),100.*correct/len(val_dataset)
    ))


#训练9次看看效果
#for epoch in range(1, 10):
    #train(model=model_ft, device=DEVICE, train_loader=image_dataloader["train"],epoch=epoch)
    #test(model=model_ft, device=DEVICE, test_loader=image_dataloader["valid"])

