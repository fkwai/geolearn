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

"""
HOW LSTM better than LR on some basins
"""
# test

wqData = waterQuality.DataModelWQ('Silica64')
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'
outName = 'Silica64-Y8090-00955-opt1'

# trainset = 'Y0010'
# testset = 'Y8090'
# outName = 'Silica64-Y0010-00955-opt1'

optT = trainset
master = basins.loadMaster(outName)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)
ns = len(siteNoLst)
# calculate error from sequence
nM = 2
modLst = ['LSTM', 'LR']
rmseMat = np.ndarray([ns, nM, 2])
corrMat = np.ndarray([ns, nM, 2])
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    rmseLSTM, corrLSTM = waterQuality.calErrSeq(dfPred[code], dfObs[code])
    dfP3 = wqLinear.loadSeq(siteNo, code, 'LR', optT=optT)
    rmseLR, corrLR = waterQuality.calErrSeq(dfP3[code], dfObs[code])
    rmseMat[k, :, :] = [rmseLSTM, rmseLR]
    corrMat[k, :, :] = [corrLSTM, corrLR]

fig, axes = plt.subplots(1, 2)
axes[0].plot(rmseMat[:, 0, 1], rmseMat[:, 1, 1], '*')
axes[0].plot([0, 8], [0, 8], '-k')
axes[0].set_xlim(0, 8)
axes[0].set_ylim(0, 8)
axes[1].plot(corrMat[:, 0, 1], corrMat[:, 1, 1], '*')
axes[1].plot([-0.1, 1], [-0.1, 1], '-k')
axes[1].set_xlim(-0.1, 1)
axes[1].set_ylim(-0.1, 1)
fig.show()

np.corrcoef(corrMat[:, 0, 0], corrMat[:, 1, 0])[0, 1]
np.corrcoef(rmseMat[:, 0, 0], rmseMat[:, 1, 0])[0, 1]


# box
for (errMat, title) in zip([rmseMat, corrMat], ['RMSE', 'Correlation']):
    dataBox = list()
    for k in range(2):
        temp = [errMat[:, i, k] for i in range(2)]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = modLst
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
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    # shortName = codePdf.loc[code]['shortName']
    for k in range(2):
        mapData = corrMat[:, 0, k]-corrMat[:, 1, k]
        axplot.mapPoint(axM[k], lat, lon, mapData, s=12)
        # axM[k].set_title(modLst[k])
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    axP = np.array([axP]) if nM == 1 else axP
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP1, dfObs = basins.loadSeq(outName, siteNo)
    rmse1, corr1 = waterQuality.calErrSeq(dfP1[code], dfObs[code])
    dfP3 = wqLinear.loadSeq(siteNo, code, 'LR', optT='Y8090')
    rmse3, corr3 = waterQuality.calErrSeq(dfP3[code], dfObs[code])
    t = dfObs.index.values
    tBar = np.datetime64('2000-01-01')
    styLst = '-*'
    # styLst = ['-', '-*']
    dfPLst = [dfP1, dfP3]
    rmseLst = [rmse1, rmse3]
    corrLst = [corr1, corr3]
    for k, dfP in enumerate(dfPLst):
        axplot.plotTS(axP[k], t, [dfP[code], dfObs[code]], tBar=tBar,
                      legLst=[modLst[k], 'obs'], styLst=styLst, cLst='br')
        # ind = np.where(~np.isnan(dfObs[code].values))
        # axplot.plotTS(axP[k], t[ind], dfObs[code].values[ind], tBar=tBar,
        #               legLst=[modLst[k], 'obs'], styLst=styLst, cLst='r')
        tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
            siteNo, rmseLst[k][0], rmseLst[k][1], corrLst[k][0], corrLst[k][1])
        axP[k].set_title(tStr)


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

for ax in figP.axes:
    ax.set_xlim(np.datetime64('2015-01-01'), np.datetime64('2020-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1988-01-01'), np.datetime64('1993-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1980-01-01'), np.datetime64('2020-01-01'))
figP.canvas.draw()
