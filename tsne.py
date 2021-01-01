from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 加载数据集
iris = load_iris()
# 共有150个例子， 数据的类型是numpy.ndarray
print(iris.data.shape)#(150,4)
# 对应的标签有0,1,2三种
print(iris.target.shape)#(150,)
# 使用TSNE进行降维处理。从4维降至2维。
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(iris.data)
print(type(tsne))
print(tsne.shape)
# 使用PCA 进行降维处理
pca = PCA().fit_transform(iris.data)
print(pca.shape)
# 设置画布的大小
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
plt.colorbar()
plt.show()