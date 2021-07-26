
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
# warnings.simplefilter('error')

dataName = 'G200N'

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore', category=RuntimeWarning)
#     DF = dbBasin.DataFrameBasin(dataName)
DF = dbBasin.DataFrameBasin(dataName)

codeLst = usgs.newC

trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

trainSet = 'rmRT20'
testSet = 'pkRT20'
# trainSet = 'B10'
# testSet = 'A10'
labelLst = ['QFPRT2C', 'QFRT2C', 'QFPT2C', 'FPRT2C']
nL = len(labelLst)
yLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet=testSet, ep=500)
    yOut = np.ndarray(yP.shape)
    for k, code in enumerate(codeLst):
        m = DF.g[:, DF.varG.index(code+'-M')]
        s = DF.g[:, DF.varG.index(code+'-S')]
        yOut[:, :, k] = yP[:, :, k]*s+m
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
matLst = [np.full([len(siteNoLst), len(codeLst), 4], np.nan) for x in labelLst]

for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            for k in range(nL):
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
                                label2=labelLst+['WRTDS'], label1=codeLabLst,
                                sharey=sharey)
    if statStr == 'Bias':
        for ax in axes:
            _ = ax.axhline(0)
    fig.show()
