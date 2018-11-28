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
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights