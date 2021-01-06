import numpy as np
from hydroDL.utils import grid
from hydroDL.post import axplot, figplot
import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

# temporal test
trainName = 'CONUSv2f1'
testName = 'CONUSv2f1'
out = trainName+'_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
predField = 'LSTM'
targetField = 'SMAP'

ds1 = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2015])
ds1.readData(var='SMAP_AM', field='SMAP')
ds1.readPred(rootOut=rootOut, out=out, field='LSTM', drMC=100)
statSigma1 = ds1.statCalSigma(field='LSTM')

ds2 = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
ds2.readData(var='SMAP_AM', field='SMAP')
ds2.readPred(rootOut=rootOut, out=out, field='LSTM', drMC=100)
statErr2 = ds2.statCalError(predField='LSTM', targetField='SMAP')
statSigma2 = ds2.statCalSigma(field='LSTM')

t = np.concatenate([ds1.time, ds2.time])
x = np.concatenate([ds1.LSTM, ds2.LSTM], axis=1)
y = np.concatenate([ds1.SMAP, ds2.SMAP], axis=1)
sigma = np.concatenate([statSigma1.sigma_mat, statSigma2.sigma_mat], axis=1)
lat = ds1.crd[:, 0]
lon = ds1.crd[:, 1]
tBar = np.datetime64('2016-04-01')


def funcMap():
    gridRMSE, uy, ux = grid.array2grid(statErr2.RMSE, lat=lat, lon=lon)
    gridCorr, uy, ux = grid.array2grid(statSigma2.sigma, lat=lat, lon=lon)
    figM, axM = plt.subplots(1, 2, figsize=(10, 4))
    axplot.mapGrid(axM[0], uy, ux, gridRMSE, vRange=[0, 0.1], cmap=plt.cm.jet)
    axM[0].set_title('Temporal Test RMSE')
    axplot.mapGrid(axM[1], uy, ux, gridCorr, vRange=[0, 0.1], cmap=plt.cm.jet)
    axM[1].set_title('Temporal Test Correlation')
    figP, axP = plt.subplots(1, 1, figsize=(15, 3))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    axplot.plotTS(axP, t, [x[iP, :], y[iP, :]], styLst='-*',
                  cLst='br', tBar=tBar, legLst=['pred', 'obs'])
    axP.fill_between(t, x[iP, :]+sigma[iP, :]*2,
                     x[iP, :]-sigma[iP, :]*2, facecolor='c', label=r'95% band')
    axP.legend(loc='upper right')


figplot.clickMap(funcMap, funcPoint)
