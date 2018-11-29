import logRegres
dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataArr, labelMat)
print(weights)
#logRegres.plotBestFit(weights)

#test sco gradient acent0
from numpy import *
#reload(logRegres)
#dataArr, labelMat = logRegres.loadDataSet()
#weights = logRegres.stocCradAscent0(array(dataArr), labelMat)
#logRegres.plotBestFit(weights)

#test sco gradient acent1
#dataArr, labelMat = logRegres.loadDataSet()
#weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
#logRegres.plotBestFit(weights)


#horse test
logRegres.multiTest()
