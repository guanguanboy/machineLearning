import logRegres
dataArr, labelMat = logRegres.loadDataSet()
print(logRegres.gradAscent(dataArr, labelMat))