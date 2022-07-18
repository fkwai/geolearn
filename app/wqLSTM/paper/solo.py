
import scipy
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


ep = 300
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
# label = 'FPRT2QC'
DF = dbBasin.DataFrameBasin('G200')
bQ = np.isnan(DF.q[:, :, 0])
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)

# LSTM comb
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, 1000)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, 1000)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

# solo models
nt, ns, nc = DF.c.shape
corrS1 = np.full([ns, nc], np.nan)
corrS2 = np.full([ns, nc], np.nan)
for k, code in enumerate(DF.varC):
    outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=300)
    yP[bQ] = np.nan
    pred1 = DF.extractSubset(yP, trainSet)
    pred2 = DF.extractSubset(yP, testSet)
    corrS1[:, k] = utils.stat.calCorr(pred1[:, :, 0], obs1[:, :, k])
    corrS2[:, k] = utils.stat.calCorr(pred2[:, :, 0], obs2[:, :, k])


# WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW2 = np.load(corrFile2)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for corr in [corrL1, corrL2, corrS1, corrS2, corrW1, corrW2]:
    corr[matRm] = np.nan


# re-order
indPlot = np.argsort(np.nanmean(corrL2, axis=0))
codeStrLst = list()
dataPlot = list()
for k in indPlot:
    code = DF.varC[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([corrL2[:, k], corrS2[:, k], corrW2[:, k]])
strLst = usgs.codeStrPlot(codeStrLst)
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=strLst)
fig.show()
