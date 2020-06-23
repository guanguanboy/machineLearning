import adaboost
import numpy as np
import pandas as pd

datMat, classLabels=adaboost.loadSimpleData()
#adaboost.showPlot(datMat, classLabels)

#如下代码，构建单层决策树
m = datMat.shape[0]
D = np.mat(np.ones((m, 1)) / m) #初始化样本权重（每个样本权重相等）
#classLabels = classLabels.reshape(m, 1)
yMat = np.array(classLabels)
yMat = yMat.reshape(m, 1)
bestStump, minE, bestClas= adaboost.get_Stump(datMat, yMat, D)
print(bestStump)
print(minE)
print(bestClas)

#训练分类器
weakClass, aggClass = adaboost.Ada_train(datMat, yMat, maxC = 40)
print(weakClass)
print(aggClass)

#基于Adaboost的分类
finalClass = adaboost.AdaClassify([0,0], weakClass)
print(finalClass)

#如下为使用Adaboost在病马数据集上的分类的代码

#载入数据
train = pd.read_table('horseColicTraining2.txt', header=None)
test = pd.read_table('horseColicTest2.txt', header=None)


adaboost.calAcc(maxC=40)

#不同弱分类器数目的各种情况下，adaboost算法预测的准确率

"""

Cycles=[1, 10, 50, 100, 500, 1000, 10000]
train_acc=[]
test_acc=[]
for maxC in Cycles:
    a, b = adaboost.calAcc(maxC)
    train_acc.append(round(a*100,2))
    test_acc.append(round(b*100, 2))

df = pd.DataFrame({'分类器数目': Cycles,
                   '训练集准确率': train_acc,
                   '测试集准确率':test_acc})
print(df)

"""
datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
trainxMat, trainyMat = adaboost.getMat('horseColicTraining2.txt')
m = trainxMat.shape[0]
weakClass, aggClass = adaboost.Ada_train(trainxMat, trainyMat, 10)

#绘制病马数据集roc曲线
adaboost.plotROC(aggClass.T, labelArr)
