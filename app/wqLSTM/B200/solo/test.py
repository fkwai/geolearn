import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib


code = '00955'
dataName = '{}-{}'.format(code, 'B200')
DF = dbBasin.DataFrameBasin(dataName)

labelLst = ['FT2QC', 'QFT2C', 'QT2C']
label = 'QT2C'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
ep = 500

matObs = DF.extractT([code])
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)

# local model
corrLst1=list()
corrLst2=list()
ypLst1=list()
ypLst2=list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    dictMaster = basinFull.loadMaster(outName)    
    yP1, ycP1 = basinFull.testModel(outName, DF=DF, testSet=trainSet, ep=ep)
    yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=ep)    
    if len(dictMaster['varY'])>1:
        yP1=yP1[:,:,1:]
        yP2=yP2[:,:,1:]
    corrL1 = utils.stat.calCorr(yP1, obs1)
    corrL2 = utils.stat.calCorr(yP2, obs2)
    ypLst1.append(yP1)
    ypLst2.append(yP2)
    corrLst1.append(corrL1)
    corrLst2.append(corrL2)


# # WRTDS
yW1 = WRTDS.testWRTDS(dataName, trainSet, trainSet, [code])
yW2 = WRTDS.testWRTDS(dataName, trainSet, testSet, [code])
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)


# global model
ep = 180
DFG = dbBasin.DataFrameBasin('NY5')
label='QFT2C'
outName = '{}-{}-{}'.format('NY5', label, trainSet)
dictMaster = basinFull.loadMaster(outName)
varY = dictMaster['varY']
yP1, ycP1 = basinFull.testModel(outName, DF=DFG, testSet=trainSet, ep=ep)
yP2, ycP2 = basinFull.testModel(outName, DF=DFG, testSet=testSet, ep=ep)
matObs = DFG.extractT(varY)
obs1 = DFG.extractSubset(matObs, trainSet)
obs2 = DFG.extractSubset(matObs, testSet)
indC=varY.index(code)
corrL1 = utils.stat.calCorr(yP1, obs1)[:,indC]
corrL2 = utils.stat.calCorr(yP2, obs2)[:,indC]


dataPlot=[corrLst1+[corrW1,corrL1],corrLst2+[corrW2,corrL2]]
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(8, 4),
    label1=['train','test'],label2=labelLst+['WRTDS','LSTM all variables']    
)
fig.suptitle(code)
fig.show()
