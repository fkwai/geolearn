
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

dataName = 'G200N'
trainSet = 'rmR20'
testSet = 'pkR20'
# trainSet = 'B10'
# testSet = 'A10'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(
    outName, DF=DF, testSet=testSet, ep=500)

# deal with mean and std
codeLst = usgs.newC
yOut = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DF.g[:, DF.varG.index(code+'-M')]
    s = DF.g[:, DF.varG.index(code+'-S')]
    data = yP[:, :, k]
    yOut[:, :, k] = data*s+m

# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
mat1 = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
mat2 = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for indS, siteNo in enumerate(siteNoLst):
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            stat = utils.stat.calStat(yOut[:, indS, indC], d2.Y[:, indS, indC])
            stat2 = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            mat1[indS, indC, 0] = stat['Bias']
            mat1[indS, indC, 1] = stat['RMSE']
            mat1[indS, indC, 2] = stat['NSE']
            mat1[indS, indC, 3] = stat['Corr']
            mat2[indS, indC, 0] = stat2['Bias']
            mat2[indS, indC, 1] = stat2['RMSE']
            mat2[indS, indC, 2] = stat2['NSE']
            mat2[indS, indC, 3] = stat2['Corr']

indT1 = np.where(~np.isnan(d1.Y[:, indS, indC]))
indT2 = np.where(~np.isnan(d2.Y[:, indS, indC]))

yOut[indT2, indS, indC]
yW[indT2, indS, indC]

# select sites
dictSiteName = 'dict{}.json'.format(dataName[:4])
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)

statStrLst = ['Bias', 'RMSE', 'NSE', 'Corr']
dataPlot = list()
labelLst = [usgs.codePdf.loc[code]['shortName'] +
            '\n'+code for code in codeLst]
for k, statStr in enumerate(statStrLst):
    temp = list()
    for ic, code in enumerate(codeLst):
        [a, b], _ = utils.rmNan([mat1[:, ic, k], mat2[:, ic, k]])
        temp.append([a, b])
    sharey = False if statStr in ['Bias', 'RMSE'] else True
    fig, axes = figplot.boxPlot(temp, widths=0.5, figsize=(12, 4),
                                label2=['LSTM', 'WRTDS'], label1=labelLst,
                                sharey=sharey)
    if statStr == 'Bias':
        for ax in axes:
            _ = ax.axhline(0)
    fig.show()

#
# DF2 = dbBasin.DataFrameBasin('G400')

# labelLst = [usgs.codePdf.loc[code]['shortName'] + code for code in codeLst]
# d1 = dbBasin.DataModelBasin(DF2, subset=trainSet, varY=codeLst)
# d2 = dbBasin.DataModelBasin(DF2, subset=testSet, varY=codeLst)
# k = 60
# dataPlot = [yW[:, k, :], d1.Y[:, k, :], d2.Y[:, k, :]]
# cLst = ['red', 'grey', 'black']
# fig, axes = figplot.multiTS(
#     DF.t, dataPlot, labelLst=labelLst, cLst=cLst)
# fig.show()

