from math import log
import operator
#创建数据集，计算经验熵

"""
函数说明：创建测试数据集
Parameters：无
Returns：
    dataSet：数据集
    labels：分类属性
Modify：
    2018-03-12

"""

def createDataSet():
    #数据集,使用列表
    dataSet =[[0, 0, 0, 0, 'no'],
              [0, 0, 0, 1, 'no'],
              [0, 1, 0, 1, 'yes'],
              [0, 1, 1, 0, 'yes'],
              [0, 0, 0, 0, 'no'],
              [1, 0, 0, 0, 'no'],
              [1, 0, 0, 1, 'no'],
              [1, 1, 1, 1, 'yes'],
              [1, 0, 1, 2, 'yes'],
              [1, 0, 1, 2, 'yes'],
              [2, 0, 1, 2, 'yes'],
              [2, 0, 1, 1, 'yes'],
              [2, 1, 0, 1, 'yes'],
              [2, 1, 0, 2, 'yes'],
              [2, 0, 0, 0, 'no']]

    #分类属性
    labels=['年龄', '有工作', '有自己的房子', '信贷情况']

    #返回数据集和分类属性
    return dataSet, labels

"""
函数说明：计算给定数据集的经验熵（香农熵）
Parameters：
    dataSet：数据集
Returns：
    shannonEnt：经验熵
Modify：
    2018-03-12

"""

def calcShannonEnt(dataSet):
    #返回数据集行数
    numEntries=len(dataSet)

    #保存每个标签（label）出现次数的字典
    labelCounts={}

    #对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel = featVec[-1]      #提取标签信息
        if currentLabel not in labelCounts.keys(): # 如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel]+=1    #label计数

    shannonEnt=0.0 #经验熵初始化

    #计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  #该标签的概率
        shannonEnt -= prob*log(prob, 2)

    return shannonEnt

#如下计算信息增益

#信息增益的定义：信息增益表示 得知特征X的信息 而 使得类Y的 信息不确定性减少的程度。
#条件熵H(Y∣X)H(Y|X)H(Y∣X)表示在已知随机变量X的条件下随机变量Y的不确定性

"""
函数说明：按照给定特征划分数据集
Parameters：
    dataSet：待划分的数据集
    axis：划分数据集的特征
    value：需要返回的特征的值
Returns：
    retDataSet：数据集划分的结果
Modify：
    2018-03-12

"""
def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:#将等于某个特征值的样本放到子集中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            # 这两行代码相当于把axis所在元素从列表中去掉了,为什么需要去掉这个特征的子集，原因是后面会对子集根据剩余的特征进行再划分
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
函数说明：计算给定数据集的信息增益
Parameters：
    dataSet：数据集
Returns：
    shannonEnt：信息增益最大特征的索引值
Modify：
    2018-03-12

"""
def chooseBestFeatureToSplit(dataSet):
    #特征数量
    numFeatures = len(dataSet[0]) - 1

    #计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)

    #信息增益
    bestInfoGain = 0.0

    #最优特征值索引值
    bestFeature = -1

    #遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet 的第i个特征的所有的值
        featList = [example[i] for example in dataSet]

        #创建set集合{}, 元素不可重复，目点是为了获取第i个特征的不可重复的值（比如，0，1，2，分别表示青年、中年、老年）
        uniqueVals = set(featList)

        #经验条件熵
        newEntropy = 0.0

        #计算信息增益
        for value in uniqueVals:

            #划分子集，基于某个特征的特定的值
            subDataSet = splitDataSet(dataSet, i, value)

            #计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))

            #根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))

        #信息增益
        infoGain = baseEntropy - newEntropy

        #打印每个特征的信息增益
        print("第 %d 个特征的增益为%.3f" % (i, infoGain))

        if (infoGain > bestInfoGain):
            #更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain

            #记录信息增益最大特征的索引值
            bestFeature = i

    return bestFeature

"""
#mian函数
if __name__=='__main__':
    dataSet, features=createDataSet()
    #print(dataSet)
    #print(calcShannonEnt(dataSet))
    print("最优索引值： " + str(chooseBestFeatureToSplit(dataSet)))
"""
"""
函数说明：统计classList中出现次数最多的元素（类标签）
Parameters：
    classList：类标签列表
Returns：
    sortedClassCount[0][0]：出现次数最多的元素（类标签）
Modify：
    2018-03-13

"""
def majorityCnt(classList):
    classCount={}

    #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1

        #根据字典的值降序排列
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #由于是降序，所以reverse为true
        #由于字典中存储的是二元组，即键、值，operator.itemgetter(1)表示根据值进行排序
        return sortedClassCount[0][0]

"""
函数说明：创建决策树
ID3算法
Parameters:
    dataSet：训练数据集
    labels：分类属性标签
    featLabels：存储选择的最优特征标签
Returns：
    myTree：决策树
Modify：
    2018-03-13

"""
def createTree(dataSet, labels, featLabels):
    #取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]

    #如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    #遍历完所有特征时，返回出现次数最多的类标签，当dataSet中一条记录的长度为1（只包含标签了）时
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    #选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)

    #最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)

    #根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}

    #删除已经使用的特征标签，删除之后labels中存储的是剩余的特征
    del(labels[bestFeat])

    #得到训练集中所有最优特征的属性值
    featValues=[example[bestFeat] for example in dataSet]

    #去掉重复的属性值
    uniqueVls=set(featValues)

    #遍历特征，创建决策树
    for value in uniqueVls:
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)

    return myTree

if __name__=='__main__':
    dataSet, labels=createDataSet()
    #print(dataSet)
    #print(calcShannonEnt(dataSet))
    #print("最优索引值： " + str(chooseBestFeatureToSplit(dataSet)))
    featLabels=[]
    myTree=createTree(dataSet, labels, featLabels)
    print(myTree)