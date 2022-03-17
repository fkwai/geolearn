
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

import warnings

DFN = dbBasin.DataFrameBasin('G200N')
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC

# trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
# testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

dataNameLst = ['G200', 'G200N']
trainSet = 'rmRT20'
testSet = 'pkRT20'
label = 'QFPRT2C'

yLst = list()
for dataName in dataNameLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet=testSet, ep=500)
    yOut = np.ndarray(yP.shape)
    if dataName[-1] == 'N':
        yP, ycP = basinFull.testModel(
            outName, DF=DFN, testSet=testSet, ep=500)
        yOut = np.ndarray(yP.shape)
        for k, code in enumerate(codeLst):
            m = DFN.g[:, DFN.varG.index(code+'-M')]
            s = DFN.g[:, DFN.varG.index(code+'-S')]
            yOut[:, :, k] = yP[:, :, k]*s+m
    else:
        yP, ycP = basinFull.testModel(
            outName, DF=DF, testSet=testSet, ep=500)
        yOut = yP
    yLst.append(yOut)


# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
matW = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
matLst = [np.full([len(siteNoLst), len(codeLst), 4], np.nan)
          for x in dataNameLst]

for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            for k, yL in enumerate(yLst):
                yL = yLst[k]
                statL = utils.stat.calStat(
                    yL[:, indS, indC], d2.Y[:, indS, indC])
                matLst[k][indS, indC, :] = list(statL.values())


# select sites
statStrLst = ['Bias', 'RMSE', 'NSE', 'Corr']
codeLabLst = [usgs.codePdf.loc[code]['shortName'] +
              '\n'+code for code in codeLst]
for k, statStr in enumerate(statStrLst):
    dataPlot = list()
    for ic, code in enumerate(codeLst):
        temp = [mat[:, ic, k] for mat in matLst] + [matW[:, ic, k]]
        temp2, _ = utils.rmNan(temp)
        dataPlot.append(temp2)
    sharey = False if statStr in ['Bias', 'RMSE'] else True
    fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                                label2=['LSTM','LSTM Norm','WRTDS'], label1=codeLabLst,
                                sharey=sharey)
    if statStr == 'Bias':
        for ax in axes:
            _ = ax.axhline(0)
    fig.show()
