
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

dataName = 'G400Norm'
outName = dataName
trainSet = 'rmRT20'
testSet = 'pkRT20'

DF = dbBasin.DataFrameBasin(outName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=200)


# deal with mean and std
codeLst = usgs.newC
yOut = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DF.g[:, DF.varG.index(code+'-M')]
    s = DF.g[:, DF.varG.index(code+'-S')]
    data = yP[:, :, k]
    yOut[:, :, k] = data*s+m

# WRTDS
yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
mat1 = np.ndarray([len(siteNoLst), len(codeLst), 4])
mat2 = np.ndarray([len(siteNoLst), len(codeLst), 4])
for indS, siteNo in enumerate(siteNoLst):
    for indC, code in enumerate(codeLst):
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

statStrLst = ['Bias', 'RMSE', 'NSE', 'Corr']
dataPlot = list()
labelLst = [usgs.codePdf.loc[code]['shortName'] +
            '\n'+code for code in codeLst]
for k, statStr in enumerate(statStrLst):
    temp = list()
    for ic, code in enumerate(codeLst):
        temp.append([mat1[:, ic, k], mat2[:, ic, k]])
    fig, axes = figplot.boxPlot(temp, widths=0.5, figsize=(12, 4),
                                label2=['LSTM', 'WRTDS'], label1=labelLst,  sharey=True)
    # for ax in axes:
    #     ax.axhline(0)
    fig.show()

#
DF2 = dbBasin.DataFrameBasin('G400')

labelLst = [usgs.codePdf.loc[code]['shortName'] + code for code in codeLst]
d1 = dbBasin.DataModelBasin(DF2, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF2, subset=testSet, varY=codeLst)
k = 60
dataPlot = [yW[:, k, :], d1.Y[:, k, :], d2.Y[:, k, :]]
cLst = ['red', 'grey', 'black']
fig, axes = figplot.multiTS(
    DF.t, dataPlot, labelLst=labelLst, cLst=cLst)
fig.show()
