
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



dataName = 'NY5'
labelLst = ['FT2QC', 'QFT2C', 'QT2C']
trainLst = ['B15', 'rmYr5b0', 'rmRT5b0']
testLst = ['A15', 'pkYr5b0', 'pkRT5b0']

label='QFT2C'
trainSet='B15'
testSet='A15'


outName = '{}-{}-{}'.format(dataName, label, trainSet)
ep=80
DF = dbBasin.DataFrameBasin(dataName)

dictMaster=basinFull.loadMaster(outName)
varY =dictMaster['varY']
yP1, ycP1 = basinFull.testModel(
                outName, DF=DF, testSet=trainSet, ep=ep)
yP2, ycP2 = basinFull.testModel(
                outName, DF=DF, testSet=testSet, ep=ep)

matObs = DF.extractT(varY)
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)

# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
folderName = '{}-{}-{}'.format(dataName, trainSet, 'all')

yWLst=list()
for k,siteNo in enumerate(DF.siteNoLst):
    print('reading {} {}'.format(k,siteNo))
    fileName=os.path.join(dirWRTDS,folderName,siteNo)
    dfW=pd.read_csv(fileName,index_col=0)
    yWLst.append(dfW.values)
yW=np.stack(yWLst,axis=-1).swapaxes(1,2)
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2]:
    corr[matRm] = np.nan


# re-order
indPlot = np.argsort(np.nanmean(corrL2, axis=0))
codeStrLst = list()
dataPlot = list()
for k in indPlot:
    code = varY[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([corrW1[:, k], corrL1[:, k]])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=codeStrLst,yRange=[0,1])
fig.show()