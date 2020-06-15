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
import scipy.signal as signal
from astropy.timeseries import LombScargle

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
    figP, axP = plt.subplots(4, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP1, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred.index.values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    # plot Q
    rmse, corr = waterQuality.calErrSeq(dfP1['00060'], dfObs['00060'])
    axplot.plotTS(axP[0], t, [dfP1['00060'], dfObs['00060']], tBar=tBar,
                  legLst=['pred', 'obs'], styLst='--', cLst='br')
    tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
        siteNo, rmse[0], rmse[1], corr[0], corr[1])
    axP[0].set_title('steamflow '+tStr)
    # Silica
    rmse, corr = waterQuality.calErrSeq(dfP1[code], dfObs[code])
    axplot.plotTS(axP[1], t, [dfP1[code], dfObs[code]], tBar=tBar,
                  legLst=['LSTM', 'obs'], styLst='-*', cLst='br')
    tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
        siteNo, rmse[0], rmse[1], corr[0], corr[1])
    axP[1].set_title('Silica '+tStr)
    # fourier
    df = dfObs[dfObs['00955'].notna().values]
    # nt = len(dfObs)
    nt = 365*5
    x = (df.index.values.astype('datetime64[D]') -
         np.datetime64('1979-01-01')).astype(np.float)
    y = df['00955'].values
    freq = np.fft.fftfreq(nt)[1:]
    ls = LombScargle(x, y)
    power = ls.power(freq)
    df2 = dfP1['00955']
    x2 = (df2.index.values.astype('datetime64[D]') -
          np.datetime64('1979-01-01')).astype(np.float)
    y2 = df2.values
    ls2 = LombScargle(x2, y2)
    power2 = ls2.power(freq)
    axP[2].set_ylabel('normalize spectrum')
    indF = np.where(freq > 0)[0]
    axP[2].plot(1/freq[indF], power2[indF], 'b', label='lstm')
    axP[2].plot(1/freq[indF], power[indF], 'r', label='obs')
    axP[2].legend()
    axP[3].set_ylabel('power')
    axP[2].set_xlabel('period (day)')
    axP[3].plot(np.log(freq), np.log(power), '-*')
    axP[3].set_xlabel('log(freq)')
    axP[3].set_ylabel('log(power)')


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
