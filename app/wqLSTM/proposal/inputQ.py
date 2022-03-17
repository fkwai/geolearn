
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

dataName = 'G200'
trainSet = 'rmR20'
testSet = 'pkR20'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = DF.varC
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)

# LSTM
labelLst = ['FPRT2C', 'FPRT2QC', 'QT2C', 'QFPRT2C']
yPLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=500)
    master = basinFull.loadMaster(outName)
    indC = [master['varY'].index(x) for x in codeLst]
    yP = yP[:, :, indC]
    yPLst.append(yP)


# correlation
corrLst1 = list()
bQ = np.isnan(DF.q[:, :, 0])
for yP in yPLst:
    yT = yP.copy()
    yT[bQ, :] = np.nan
    corr = utils.stat.calCorr(DF.extractSubset(yT, testSet), obs2)
    corrLst1.append(corr)
corrLst2 = list()
bQ = np.isnan(DF.q[:, :, 0])
for yP in yPLst:
    yT = yP.copy()
    yT[bQ, :] = np.nan
    corr = utils.stat.calCorr(DF.extractSubset(yT, testSet),
                              DF.extractSubset(yW, testSet))
    corrLst2.append(corr)
corrW = utils.stat.calCorr(DF.extractSubset(yW, testSet), obs2)
# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 160) & (count2 < 40)
for corr in corrLst1:
    corr[matRm] = np.nan
for corr in corrLst2:
    corr[matRm] = np.nan


# box plot - all cases
dataPlot = list()
# codePlot = [codeLst[k] for k in np.argsort(np.nanmean(matLR, axis=0))]
codePlot = ['00935', '00955', '00940', '00945',
            '00930', '00095', '00915', '00925']
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codePlot]
for code in codePlot:
    ic = codeLst.index(code)
    # dataPlot.append([corr[:, ic] for corr in corrLst1]+[corrW[:, ic]])
    dataPlot.append([corr[:, ic] for corr in corrLst1])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label1=codeStrLst, cLst='rgbk',
                            label2=['F-C','F-QC', 'Q-C', 'FQ-C'])
fig.show()

