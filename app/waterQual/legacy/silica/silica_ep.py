import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# test
outName = 'Silica64-Y8090-00955-opt1'

wqData = waterQuality.DataModelWQ('Silica64')
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'
# trainset = 'Y0010'
# testset = 'Y8090'
optT = trainset
master = basins.loadMaster(outName)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
epLst = [100, 200, 300, 400, 500]
epLst = [100,  300,  500]
for ep in epLst:
    basins.testModelSeq(outName, siteNoLst, wqData=wqData, ep=ep)
ns = len(siteNoLst)
nep = len(epLst)
# calculate error from sequence
rmseMat = np.ndarray([ns, 5, 2])
corrMat = np.ndarray([ns, 5, 2])
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    for i, ep in enumerate(epLst):
        dfPred, dfObs = basins.loadSeq(outName, siteNo, ep=ep)
        rmseLSTM, corrLSTM = waterQuality.calErrSeq(dfPred[code], dfObs[code])
        rmseMat[k, i, :] = rmseLSTM
        corrMat[k, i, :] = corrLSTM

# box
for (errMat, title) in zip([rmseMat, corrMat], ['RMSE', 'Correlation']):
    dataBox = list()
    for k in range(2):
        temp = [errMat[:, i, k] for i in range(nep)]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = epLst
    fig = figplot.boxPlot(dataBox, label1=label1, label2=label2, sharey=True)
    fig.suptitle(title)
    fig.show()


# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    nM = nep
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    axM = np.array([axM]) if nM == 1 else axM
    shortName = codePdf.loc[code]['shortName']
    title = '{} {}'.format(shortName, code)
    for k in range(nM):
        axplot.mapPoint(axM[k], lat, lon, corrMat[:, k, 1], s=12)
        axM[k].set_title(epLst[k])
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    axP = np.array([axP]) if nM == 1 else axP
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPLst = list()
    rmseLst = list()
    corrLst = list()
    for ep in epLst:
        dfP, dfObs = basins.loadSeq(outName, siteNo, ep=ep)
        rmse, corr = waterQuality.calErrSeq(dfP[code], dfObs[code])
        dfPLst.append(dfP)
        rmseLst.append(rmse)
        corrLst.append(corr)
    t = dfObs.index.values
    tBar = np.datetime64('2000-01-01')
    for k, dfP in enumerate(dfPLst):
        axplot.plotTS(axP[k], t, [dfP[code], dfObs[code]], tBar=tBar,
                      legLst=[epLst[k], 'obs'], styLst='-*', cLst='br')
        tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
            siteNo, rmseLst[k][0], rmseLst[k][1], corrLst[k][0], corrLst[k][1])
        axP[k].set_title(tStr)


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

for ax in figP.axes:
    ax.set_xlim(np.datetime64('2010-01-01'), np.datetime64('2015-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1990-01-01'), np.datetime64('1995-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1980-01-01'), np.datetime64('2020-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_ylim(5, 30)
figP.canvas.draw()
