from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []

    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split() #strip 的功能是删除开头或者结尾处的指定字符，如果strip 不带参数，表示删除空白符（包括'\n', '\r', '\t',  ' ')
        #split的功能是进行字符串分割，如果不指定参数，表示按照空白字符进行删除
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #将X0的值设置为1.0
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #convert to Numpy matrix
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1)) #生成一个包含n行1列的全1数组
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights

def stocCradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n) #生成一个包含5个元素的全1数组
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #dataMatrix[i]表示取出i行数据，也就是第ige样本的数据
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]

    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    #weights = wei.getA() #getA的功能是将matrix 转换为ndarray类型，因为只有ndarray类型可以直接索引
    weights = wei #测试stocCradAscent0时使用
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    print(dataArr)
    print(shape(dataArr))
    n = shape(dataArr)[0]
    print(n)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    print(x)
    y = (-weights[0] - weights[1]*x) / weights[2]
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m)) #生成一个从0到m-1的列表，一般用在循环中
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex))) #random.uniform方法将随机生成一个实数，范围在[0, len(dataIndex))之间，但是不包含len(dataIndex)
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #删除列表中样本的索引值 ,这里会有一个问题，仅仅删除索引值，没有删除矩阵中的样本，有可能导致样本被重复选中

    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if(prob > 0.5):
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line .strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = float(errorCount) / numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()

    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))