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


trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
label = 'QFT2C'
hucLst = range(1, 18)
code = '00915'
epG = 500
epR = 500
statFunc=utils.stat.calBias

dataName = '{}-{}'.format(code, 'B200')
DF = dbBasin.DataFrameBasin(dataName)
DFA = dbBasin.DataFrameBasin('rmTK-B200')

# load global solo model
outName = '{}-{}-{}'.format(dataName, label, trainSet)
dictMaster = basinFull.loadMaster(outName)
yP1, ycP1 = basinFull.testModel(outName, DF=DF, testSet=trainSet, ep=epG)
yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=epG)
if len(dictMaster['varY']) > 1:
    yP1 = yP1[:, :, 1:]
    yP2 = yP2[:, :, 1:]

# load WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('rmTK-B200', trainSet, 'all.npz')
yWA = np.load(os.path.join(dirWRTDS, fileName))['yW']
yWA1 = DFA.extractSubset(yWA, trainSet)
yWA2 = DFA.extractSubset(yWA, testSet)
iC=usgs.varC.index(code)
indSA = [DFA.siteNoLst.index(siteNo) for siteNo in DF.siteNoLst]

yW1=yWA1[:,indSA,iC]
yW2=yWA2[:,indSA,iC]

# regional model
huc = 12
epR = 300

dataPlot1=list()
dataPlot2=list()

for huc in list(range(1,5))+list(range(6,18)):
    trainRegion = '{}_HUC{:02d}'.format(trainSet, huc)
    testRegion = '{}_HUC{:02d}'.format(testSet, huc)
    outName = '{}-{}-{}'.format(dataName, label, trainRegion)
    yR1, ycR1 = basinFull.testModel(outName, DF=DF, testSet=trainRegion, ep=epR)
    yR2, ycR2 = basinFull.testModel(outName, DF=DF, testSet=testRegion, ep=epR)
    obsR1 = DF.extractSubset(DF.c, trainRegion)
    obsR2 = DF.extractSubset(DF.c, testRegion)
    yWR1=DF.extractSubset(yW1[:,:,None],trainRegion)
    yWR2=DF.extractSubset(yW2[:,:,None],testRegion)
    yPR1=DF.extractSubset(yP1,trainRegion)
    yPR2=DF.extractSubset(yP2,testRegion)
    sR1=statFunc(yR1,obsR1)
    sR2=statFunc(yR2,obsR2)
    sW1=statFunc(yWR1,obsR1)
    sW2=statFunc(yWR2,obsR2)
    sP1=statFunc(yPR1,obsR1)
    sP2=statFunc(yPR2,obsR2)
    dataPlot1.append([sR1,sW1,sP1])
    dataPlot2.append([sR2,sW2,sP2])

fig,axes=figplot.boxPlot(dataPlot1, label2=['Region','WRTDS','Global'])
fig.show()
fig,axes=figplot.boxPlot(dataPlot2, label2=['Region','WRTDS','Global'])
fig.show()