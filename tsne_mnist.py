import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits

from sklearn.manifold import TSNE

digits = load_digits(n_class=6)
X = digits.data#1083*64
y = digits.target#1083 max(y)=5
print(X.shape)
print(y.shape)

n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

#按行放置 取前400个图片
#img中会有0、10、20列这些没有，是为了形成空列吧。
plt.imshow(img, cmap=plt.cm.binary)#cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
plt.title('A selection from the 64-dimensional digits dataset')
#plt.show()

import time#需导入包
tsne = TSNE(n_components=3, init='pca', random_state=0)
t0 = time
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)

def plot_embedding_3d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i,2],str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)

plot_embedding_3d(X_tsne,"t-SNE 3D " )
plt.show()

import time#需导入包
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y)
plt.colorbar()

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

plot_embedding(X_tsne,y,"t-SNE 2D")
plt.show()