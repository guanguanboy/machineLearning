import torch
import torch.nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

print(torch.__version__)

#生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1) ##按维数1（列）拼接
print(poly_features.size())

labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#定义、训练和测试模型
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()

num_epochs, loss = 100, torch.nn.MSELoss()
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    #定义模型，这是一个线性模型，
    print('train_features.shape = ', train_features.shape)
    print('train_features.shape[-1]', train_features.shape[-1]) #最后一维的维数，这里是列数
    print('train_features.shape[-2]', train_features.shape[-2]) #倒数第二维的维数，这里是行数

    net = torch.nn.Linear(train_features.shape[-1], 1)

    batch_size = min(10, train_labels.shape[0]) #train_labels.shape[0]表示train_labels第一维大小
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)
#三阶多项式函数拟合（正常）
#fit_and_plot(poly_features[:n_train, :], poly_features[n_train:,:],
              #labels[:n_train], labels[n_train:])

#线性函数拟合（欠拟合）
#fit_and_plot(features[:n_train,:], features[n_train:, :], labels[:n_train],labels[n_train:])

#训练样本不足（过拟合）
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:])
