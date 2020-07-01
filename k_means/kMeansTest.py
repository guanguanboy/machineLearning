import kMeans
from numpy import *

dataMat = mat(kMeans.loadDataSet('testSet.txt'))

print(min(dataMat[:,0]))
print(min(dataMat[:,1]))
print(max(dataMat[:,0]))
print(max(dataMat[:,1]))

print(kMeans.randCent(dataMat,2))
print(kMeans.distEclud(dataMat[0],dataMat[1]))

#myCentroids, clustAssing = kMeans.kMeans(dataMat,4)
"""
print("result myCentroids")
print(myCentroids)
print("result clustAssing")
print(clustAssing)
"""
dataMat3 = mat(kMeans.loadDataSet('testSet2.txt'))
centList, myNewAssements=kMeans.biKmeans(dataMat3, 3)

print(centList)