import adaboost
import numpy as np

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
