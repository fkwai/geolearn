from hydroDL.app.waterQuality import WRTDS
from hydroDL.data import usgs

codeLst=['00618','00915','00955']
trainLst = ['rmYr5b{}'.format(k) for k in range(5)] + ['B15']
testLst =  ['pkYr5b{}'.format(k) for k in range(5)] + ['A15'] 

for code in codeLst:    
    dataName='{}-{}'.format(code,'B200')
    for trainSet, testSet in zip(trainLst, testLst):
        yW = WRTDS.testWRTDS(dataName, trainSet, testSet, [code], logC=True)
        yW = WRTDS.testWRTDS(dataName, trainSet, testSet, [code], logC=False)