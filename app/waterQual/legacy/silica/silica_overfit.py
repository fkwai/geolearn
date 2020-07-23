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
basins.testModelSeq(outName, siteNoLst, wqData=wqData)
ns = len(siteNoLst)
# calculate error from sequence
rmseMat = np.ndarray([ns, 2])
corrMat = np.ndarray([ns, 2])
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    rmseLSTM, corrLSTM = waterQuality.calErrSeq(dfPred[code], dfObs[code])
    rmseMat[k, :] = rmseLSTM
    corrMat[k, :] = corrLSTM

# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    axplot.mapPoint(axM[0], lat, lon, corrMat[:, 0]-corrMat[:, 1], s=12)
    axplot.mapPoint(axM[1], lat, lon, corrMat[:, 1], s=12)
    figP, axP = plt.subplots(1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP1, dfObs = basins.loadSeq(outName, siteNo)
    rmse1, corr1 = waterQuality.calErrSeq(dfP1[code], dfObs[code])
    t = dfObs.index.values
    tBar = np.datetime64('2000-01-01')
    axplot.plotTS(axP, t, [dfP1[code], dfObs[code]], tBar=tBar,
                  legLst=['LSTM', 'obs'], styLst='-*', cLst='br')
    tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
        siteNo, rmse1[0], rmse1[1], corr1[0], corr1[1])
    axP.set_title(tStr)


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
