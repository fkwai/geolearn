import scipy
import importlib
from hydroDL import pathSMAP, master, utils
from hydroDL.master import default
from hydroDL.post import axplot, stat, figplot
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

# test
subset = 'CONUSv2f1'
out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_DA2015')
df1, yf1, obs1 = master.test(
    out, tRange=[20150402, 20180401], subset=subset, batchSize=100)
df2, yf2, obs2 = master.test(
    out, tRange=[20160401, 20180401], subset=subset, batchSize=100)
out = os.path.join(pathSMAP['Out_L3_NA'], 'DA', 'CONUSv2f1_LSTM2015')
df1, yp1, obs1 = master.test(
    out, tRange=[20150402, 20180401], subset=subset, batchSize=100)
df2, yp2, obs2 = master.test(
    out, tRange=[20160401, 20180401], subset=subset, batchSize=100)

statF = stat.statError(yf2.squeeze(), obs2.squeeze())
statP = stat.statError(yp2.squeeze(), obs2.squeeze())
t = df1.getT()
lat, lon = df1.getGeo()
dataTS = [obs1.squeeze(), yp1.squeeze(), yf1.squeeze()]
tBar = np.datetime64('2016-04-01')


def funcMap():
    gridF, uy, ux = utils.grid.array2grid(statF['RMSE'], lat=lat, lon=lon)
    gridP, uy, ux = utils.grid.array2grid(statP['RMSE'], lat=lat, lon=lon)
    figM, axM = plt.subplots(1, 2, figsize=(10, 4))
    axplot.mapGrid(axM[0], uy, ux, gridF, vRange=[0, 0.1], cmap=plt.cm.jet)
    axM[0].set_title('Temporal Test RMSE of LSTM-DI')
    axplot.mapGrid(axM[1], uy, ux, gridP, vRange=[0, 0.1], cmap=plt.cm.jet)
    axM[1].set_title('Temporal Test RMSE of LSTM')
    figP, axP = plt.subplots(1, 1, figsize=(15, 3))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    axplot.plotTS(axP, t, [data[iP, :] for data in dataTS], styLst='*--',
                  cLst='kbr', tBar=tBar, legLst=['obs', 'LSTM', 'LSTM-DI'])


figplot.clickMap(funcMap, funcPoint)
