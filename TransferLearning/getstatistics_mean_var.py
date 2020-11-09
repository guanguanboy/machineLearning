import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import os

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    #train_dataset = datasets.ImageFolder(root=r'D:\cifar10_images\test', transform=get_transform_for_train())
    data_dir = 'Datasets/datasets_detcting'
    # 加载数据这部分需要重新编写代码，使其可以加载我们自己的图片
    # Compose函数将图片的几种变换组合起来
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(),  # 将图片变成灰度图片
            transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # 缩放图片，保持长宽比不变，最短边的长为256像素,
            transforms.CenterCrop(224),  # 从中间切出 224*224的图片
            # transforms.Grayscale(),#将图片变成灰度图片
            transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]

            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    print(getStat(image_datasets['train']))
    print(getStat(image_datasets['val']))