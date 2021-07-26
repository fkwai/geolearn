
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
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

trainSet = 'rmR20'
testSet = 'pkR20'
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
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

code = '00945'
indC = codeLst.index(code)

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
matW = np.full([len(siteNoLst), 4], np.nan)
matLst = [np.full([len(siteNoLst),  4], np.nan) for x in labelLst]
for indS, siteNo in enumerate(siteNoLst):
    n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
    n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
    if n1 >= 160 and n2 >= 40:
        statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
        matW[indS, :] = list(statW.values())
        for k in range(nL):
            yL = yLst[k]
            statL = utils.stat.calStat(
                yL[:, indS, indC], d2.Y[:, indS, indC])
            matLst[k][indS, :] = list(statL.values())

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
figM, axM = plt.subplots(nL+1, 1, figsize=(8, 6))
for k, label in enumerate(labelLst):
    axplot.mapPoint(axM[k], lat, lon, matLst[k][:, -1], s=12)
axplot.mapPoint(axM[-1], lat, lon, matW[:, -1], s=12)
figM.show()


code = '00955'
indC = codeLst.index(code)
indS = 0
figP, axP = plt.subplots(1, 1, figsize=(12, 3))
dataTS = [y[:, indS, indC] for y in yLst[:3]] + \
    [DF.c[:, indS, indC]]
# dataTS = [yLst[2][:, indS, indC], yLst[1][:, indS, indC]] + \
#     [yW[:, indS, indC]]+[DF.c[:, indS, indC]]
axplot.plotTS(axP, DF.t, dataTS, cLst='bcgk')
figP.show()


def funcMap():
    nM = len(plotVar)
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    axM = np.array([axM]) if nM == 1 else axM
    for k, var in enumerate(plotVar):
        if var == '00060':
            axplot.mapPoint(axM[k], lat, lon, errMatQ2[:, 0, 1], s=12)
            axM[k].set_title('streamflow')
        else:
            ic = master['varYC'].index(var)
            shortName = codePdf.loc[var]['shortName']
            title = '{} {}'.format(shortName, var)
            axplot.mapPoint(axM[k], lat, lon, errMatC2[:, ic, 1], s=12)
            axM[k].set_title(title)
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    axP = np.array([axP]) if nM == 1 else axP
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred.index.values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')

    info1 = wqData.subsetInfo(trainset)
    info2 = wqData.subsetInfo(testset)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    tp = np.concatenate([t1, t2])
    yp = np.concatenate([ycP1[ind1], ycP2[ind2]])

    for k, var in enumerate(plotVar):
        rmse, corr = waterQuality.calErrSeq(dfPred[var], dfObs[var])
        tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
            siteNo, rmse[0], rmse[1], corr[0], corr[1])
        if var == '00060':
            styLst = '--'
            title = 'streamflow '+tStr
            axplot.plotTS(axP[k], t, [dfPred[var], dfObs[var]], tBar=tBar,
                          legLst=['LSTM', 'observation'], styLst=styLst, cLst='br')
            axP[k].set_title(title)
        else:
            styLst = '-*'
            shortName = codePdf.loc[var]['shortName']
            title = shortName + ' ' + tStr
            axplot.plotTS(axP[k], t, dfPred[var], tBar=tBar,
                          legLst=['LSTM-sequence'], styLst='-', cLst='b')
            axplot.plotTS(axP[k], tp, yp, legLst=[
                'LSTM-sample'], styLst='*', cLst='g')
            axplot.plotTS(axP[k], t, dfObs[var],
                          legLst=['observation'], styLst='*', cLst='r')
            axP[k].set_title(title)


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)
