import svmMLiA
from numpy import *
import matplotlib.pyplot as plt

def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集
    Returns:
        wc  回归系数
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))

    # w= Σ[1~n] ai*yi*xi
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotfig_SVM(xMat, yMat, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = mat(xMat)
    yMat = mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()




dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')

b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

# 画图
ws = calcWs(alphas, dataArr, labelArr)
plotfig_SVM(dataArr, labelArr, ws, b, alphas)

# print(b)
#
# print(alphas)
#
# c = alphas[alphas > 0]
#
# print(c)
# print(shape(c))
#
# for i in range(100):
#     if alphas[i] > 0.0:
#         print(dataArr[i], labelArr[i])
#
# import matplotlib.pyplot as plt
# dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
# dataArr = array(dataArr)
#
# n = shape(dataArr)[0]
#
# xcord1 = []
# ycord1 = []
# xcord2 = []
# ycord2 = []
# for i in range(n):
#     if int(labelArr[i]) == 1:
#         xcord1.append(dataArr[i,0])
#         ycord1.append(dataArr[i, 1])
#     else:
#         xcord2.append(dataArr[i, 0])
#         ycord2.append(dataArr[i, 1])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
# ax.scatter(xcord2, ycord2, s=30, c='green')
# #x = arange(-3.0, 3.0, 0.1)
# #y = (-weights[0] - weights[1] * x) / weights[2]
# #ax.plot(x, y)
#
# # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
# #x = arange(-1.0, 10.0, 0.1)
#
# # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
# #y = (-b - ws[0, 0] * x) / ws[1, 0]
# #ax.plot(x, y)
#
# # 找到支持向量，并在图中标红
# for i in range(100):
#     if alphas[i] > 0.0:
#         ax.plot(dataArr[i, 0], dataArr[i, 1], 'bo')
#
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()