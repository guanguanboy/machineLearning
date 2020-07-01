from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = map(float, curLine) #map all elements to float()
        lineArr = []
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    print(type(dataMat))
    print(shape(dataMat))
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    print('dimensions = %d' % n)
    centroids = mat(zeros((k,n))) # create centroid mat
    for j in range(n): #create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #这里发生了broadcasting，等号右侧是一个k行一列的mat
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #确定数据集中数据点的总数
    print('m=%d'% m)
    clusterAssment = mat(zeros((m,2))) # create mat to assign data points
                                        # to a centroid, also holds SE of each point
    #簇分配结果矩阵clusterAssment包含两列，一列记录簇索引值，第二列存储误差。这里的误差
    #是指当前点到簇质心的就，后边会使用该误差来评价聚类的效果。
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged: #直到数据点的分配结果不再改变
        clusterChanged = False
        for i in range(m): #for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex: #clusterAssment的初始值为0
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k): # recalculate centroids，遍历所有质心，并更新他们的取值
            pstInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]] #get all the point in this cluster，通过数据过滤来获得给定簇的所有点
            #.A是将矩阵转换为ndarray数组；nonzero函数返回的是一个元组,[0]表示取元组的第一部分
            centroids[cent,:] = mean(pstInClust, axis=0) # assign centroid to mean，沿着矩阵的列方向计算均值
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0] #数据点个数
    clusterAssment = mat(zeros((m,2))) #来存储数据集中每个点的簇分配结果及平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0] #计算整个数据集的质心
    centList = [centroid0] # create a list with one centroid，使用一个列表来保存所有的质心
    for j in range(m): #计算初始的错误值
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while(len(centList) < k):#不停的对簇进行划分，直到得到想要的簇数目为止
        lowestSSE = inf
        for i in range(len(centList)): #len(centList)是当前簇的数目， 这个for循环用来编历每一个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0], :] # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2, distMeas) #将ptsInCurrCluster这个簇分成两个簇
            sseSplit = sum(splitClustAss[:,1]) #计算将ptsInCurrCluster簇分成两个簇之后的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0], 1]) #剩余数据集的误差
            print('sseSplit = %f, and notSplit = %f' % (sseSplit, sseNotSplit))
            if (sseSplit + sseNotSplit) < lowestSSE: #将划分后的误差与没有划分的误差lowestSSE进行比较，如果划分后的误差小，则保存这个划分
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList) # change 1 to 3, 4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ')
        print(bestCentToSplit)
        print('the len of bestClustAss is: %d' % len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss #reassign new clusters, and SSE
    return mat(centList), clusterAssment


