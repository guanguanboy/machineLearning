import svmMLiA
from numpy import *
#def plotSactter():
import matplotlib.pyplot as plt
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
dataArr = array(dataArr)

n = shape(dataArr)[0]

xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
for i in range(n):
    if int(labelArr[i]) == 1:
        xcord1.append(dataArr[i,0])
        ycord1.append(dataArr[i, 1])
    else:
        xcord2.append(dataArr[i, 0])
        ycord2.append(dataArr[i, 1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
ax.scatter(xcord2, ycord2, s=30, c='green')
#x = arange(-3.0, 3.0, 0.1)
#y = (-weights[0] - weights[1] * x) / weights[2]
#ax.plot(x, y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()