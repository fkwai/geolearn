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
testName = 'CONUSv2f2'
out = trainName+'_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['Out_L3_NA']
predField = 'LSTM'
targetField = 'SMAP'
ds = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2015])
ds.readData(var='SMAP_AM', field='SMAP')
ds.readPred(rootOut=rootOut, out=out, field='LSTM')
statErr = ds.statCalError(predField='LSTM', targetField='SMAP')

lat = ds.crd[:, 0]
lon = ds.crd[:, 1]
gridRMSE, uy, ux = grid.array2grid(statErr.RMSE, lat=lat, lon=lon)
gridCorr, uy, ux = grid.array2grid(statErr.rho, lat=lat, lon=lon)
fig, axes = plt.subplots(2, 1, figsize=(9, 10))
axplot.mapGrid(axes[0], uy, ux, gridRMSE, vRange=[0, 0.1], cmap=plt.cm.jet)
axes[0].set_title('Spatial Test RMSE')
axplot.mapGrid(axes[1], uy, ux, gridCorr, vRange=[0.5, 1], cmap=plt.cm.jet)
axes[1].set_title('Spatial Test Correlation')
fig.show()

np.nanmean(statErr.RMSE)
np.nanmean(statErr.rho)
