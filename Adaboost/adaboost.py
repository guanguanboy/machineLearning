import numpy as np
import matplotlib.pyplot as plt
from numpy import *

def loadSimpleData():
    datMat = matrix(
        [
            [1., 2.1],
            [1.5, 1.6],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]
        ]
    )
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels

def showPlot(xMat, yMat):
    x = np.array(xMat[:, 0])
    y = np.array(xMat[:, 1])
    label = np.array(yMat)
    label = label.reshape(5,1)
    plt.scatter(x, y, c=label)
    plt.title('单层决策树测试数据')
    plt.show()

"""
函数功能：单层决策分类函数
参数说明：
    xMat:数据矩阵
    i:第i列，也就是第几个特征
    Q:阈值
    S:标志
返回：
    re:分类结果
"""
def classify0(xMat,i, Q, S):
    re = np.ones((xMat.shape[0],1)) #初始化re为1
    if S == 'lt':
        re[xMat[:,i] <= Q] = -1 #如果小于阈值，则赋值为-1
    else:
        re[xMat[:,i] > Q] = -1 #如果大于阈值，则赋值为-1
    return re

"""
函数功能：找到数据集上最佳的单层决策树
参数说明：
    xMat:特征矩阵
    yMat:标签矩阵
    D:样本权重
返回:
bestStump:最佳单层决策树信息
minE: 最小误差
bestClass:最佳分类结果
"""
def get_Stump(xMat, yMat, D):
    m,n = xMat.shape #m为样本个数，n为特征数
    #yMat = yMat.reshape(m,1)
    Steps = 10 #初始化一个步数
    bestStump = {} #用字典来存储树桩信息
    bestClas = np.mat(np.zeros((m,1))) #初始化分类结果为1
    minE = np.inf #最小误差初始化为正无穷大
    for i in range(n): #遍历所有特征
        Min = xMat[:,i].min() #找到特征i中最小值
        Max = xMat[:,i].max() #找到特征i中最大值
        stepSize = (Max - Min) / Steps #计算步长
        for j in range(-1, int(Steps) + 1):
            for S in ['lt', 'gt']:  #大于和小于的情况，均遍历。lt:less than, gt:greater than
                Q = (Min + j * stepSize) #计算阈值
                re = classify0(xMat, i, Q, S) #计算分类结果
                err = np.mat(np.ones((m, 1))) #初始化误差矩阵
                err[re == yMat] = 0 #分类正确的，赋值为0
                eca = D.T * err     #计算误差
                if eca < minE:
                    minE = eca
                    bestClas = re.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas

#实现完整版adaboost算法
"""
函数功能：
基于单层决策树的AdaBoost训练过程
参数说明：
    xMat:特征矩阵
    yMat:标签矩阵
    maxC:最大迭代次数
返回：
    weakClass:弱分类器信息
    aggClass:类别估计值（其实就是更改了标签的估计值）
"""
def Ada_train(xMat, yMat, maxC = 40):
    weakClass = []
    m = xMat.shape[0]
    D = np.mat(np.ones((m,1))/m) #初始化权重
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(maxC):
        Stump, error, bestClas = get_Stump(xMat, yMat, D) #构建单层决策树
        alpha = float(0.5 * np.log((1 - error)/max(error, 1e-16))) #计算弱分类器权重alpha
        Stump['alpha'] = np.round(alpha, 2) #存储弱学习算法权重，保留两位小数
        weakClass.append(Stump) #存储单层决策树
        expon = np.multiply(-1*alpha*yMat, bestClas) #计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum() #根据样本权重公式，更新样本权重
        aggClass += alpha * bestClas #更新类及类别估计值
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m,1))) #计算误差
        errRate = aggErr.sum() / m

        if errRate == 0:break #误差为0， 退出循环
    return weakClass, aggClass

#基于Adaboost的分类
"""
这里我们使用弱分类器的加权求和来计算最后的结果
函数功能：AdaBoost分类函数
参数说明：
    data:待分类样例
    classifys:训练好的分类器
    
返回：
    分类结果
"""

def AdaClassify(data, weakClass):
    dataMat = np.mat(data)
    m = dataMat.shape[0]
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)): #遍历所有分类器，进行分类
        classEst = classify0(dataMat,
                             weakClass[i]['特征列'],
                             weakClass[i]['阈值'],
                             weakClass[i]['标志'])
        aggClass += weakClass[i]['alpha'] * classEst
    return np.sign(aggClass)

