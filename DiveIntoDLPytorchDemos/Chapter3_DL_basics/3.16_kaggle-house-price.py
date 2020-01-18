import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

#获取和读取数据集
train_data = pd.read_csv('./datasets/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('./datasets/house-prices-advanced-regression-techniques/test.csv')

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]) #查看前4个样本的前4个特征、后两个特征和标签(SalePrice)

#将所有训练和测试数据的79个特征按照样本连结
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) #1:-1索引时是不包含-1的

print(all_features.shape)


#预处理数据
#针对一个连续数值的特征，求特征的均值与方差，然后每个值减去均值，除以方差，对于缺失的特征值，将其替换成该特征的均值
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index #获取连续数值特征的索引
print(numeric_features.shape)
print(numeric_features)

all_features[numeric_features] = all_features[numeric_features].apply(lambda  x: (x-x.mean())/ x.std())

#用0替换缺失值，标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)


#将离散数值转换成指示特征。举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并
#新加两个特征MSZoning_RL and MSZoning_RM，其值为0或者1.如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且
#MSZoning_RM=0.
all_features = pd.get_dummies(all_features, dummy_na=True) #dummy_na=True 会将缺失值也当作合法的特征值并为其创建指示特征,利用pandas实现one hot encode的方式
print(all_features.shape)

#通过values属性得到NumPy格式的数据，并转成Tensor方便后面训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values).view(-1, 1)

#train model
#使用一个基本的线性回归模型和平方损失函数来训练
loss = torch.nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

#定义⽐赛⽤来评价模型的对数均⽅根误差
def log_rmse(net, features, lables):
    with torch.no_grad():
        #将小于1的值设置成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        clipped_preds = clipped_preds.float()
        lables = lables.float()
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), lables.log()).mean())
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    #使用Adam
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

#K折交叉验证
"""
在K 折交叉验证中，我们把原始训练
数据集分割成K 个不重合的⼦数据集。然后我们做K 次模型训练和验证。每⼀次，我们使⽤⼀
个⼦数据集验证模型，并使⽤其他K -1 个⼦数据集来训练模型。在这K 次训练和验证中，每
次⽤来验证模型的⼦数据集都不同。最后，我们对这K 次训练误差和验证误差分别求平均。
"""
def get_k_fold_data(k, i, X, y):
    #返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k  #双斜杠表示整数除法
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #返回一个切片对象
        X_part, y_part = X[idx, :], y[idx]
        if j == i: #只有当j==i时，当这一条记录作为验证记录
            X_valid, y_valid = X_part, y_part
        elif X_train is None: #当X_train为空时，直接赋值
            X_train, y_train = X_part, y_part
        else: #当X_train不为空时，在后面追加记录
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid

def k_flod(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        train_features, train_labels, test_features, test_labels = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net,train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs+1), train_ls, 'epoch', 'rmse',
                         range(1, num_epochs+1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k  #这K 次训练误差和验证误差分别求平均，返回的是K次训练误差和验证误差的平均值


#模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_flod(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
"""
有时候你会发现⼀组参数的训练误差可以达到很低，但是在K 折交叉验证上的误差可能反而较
⾼。这种现象很可能是由于过拟合造成的。因此，当训练误差降低时，我们要观察K 折交叉验证
上的误差是否也相应降低。
"""
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


#预测，并在kaggle上提交结果
#下面定义预测函数，在预测之前，我们会使用完整的训练数据集来重新训练模型，并将预测结果存成提交所需要的格式
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1) #axis=1表示在行上操作
    submission.to_csv('./submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)


