from hydroDL.data import dbBasin
import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
# DF = dbBasin.DataFrameBasin('allCQ')

# count - only for C now
# saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
# npz = np.load(saveFile)
# matC = npz['matC']
# matCF = npz['matCF']
# matQ = npz['matQ']
# tR = npz['tR']
# codeLst = list(npz['codeLst'])
# siteNoLst = list(npz['siteNoLst'])

data = DF.c.copy()
mean = np.nanmean(data, axis=0)
std = np.nanstd(data, axis=0)
matB = ~np.isnan(data)
count = np.sum(matB, axis=0)
codeLst = DF.varC

out = (data-mean)/std

# verify
indC = DF.varC.index('00915')
indS = np.where(count[:, indC] > 200)[0]
k = 10
fig, axes = plt.subplots(2, 1)
axplot.plotTS(axes[0], DF.t, data[:, indS[k], indC])
axplot.plotTS(axes[1], DF.t, out[:, indS[k], indC])
fig.show()

# plot mean and std
code = '00915'
the = 10
bins = 20
indC = codeLst.index(code)
indS = np.where(count[:, indC] > the)[0]
fig, axes = plt.subplots(3, 2)
x = mean[indS, indC]
y = std[indS, indC]
_ = axes[0, 0].hist(x, bins=bins, density=True)
_ = axes[1, 0].hist(y, bins=bins, density=True)
axes[2, 0].plot(x, y, '*')
_ = axes[0, 1].hist(np.log(x+1), bins=bins, density=True)
_ = axes[1, 1].hist(np.log(y+1), bins=bins, density=True)
axes[2, 1].plot(np.log(x+1), np.log(y+1), '*')
fig.suptitle(code)
fig.show()

# plot all code
nfy, nfx = (5, 4)
the = 100
bins = 50

fig, axes = plt.subplots(5, 4)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, nfy, nfx)
    indC = codeLst.index(code)
    indS = np.where(count[:, indC] > the)[0]
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {} {}'.format(code, shortName, len(indS))
    dataPlot = data[:, indS, indC].flatten()
    # dataPlot = mean[indS, indC]
    # dataPlot = np.log(mean[indS, indC]+1e-5)
    # dataPlot = std[indS, indC]
    # dataPlot = np.log(std[indS, indC]+1e-5)
    _ = axes[j, i].hist(dataPlot, bins=bins, density=True)
    axplot.titleInner(axes[j, i], titleStr)
fig.show()

# all code mean vs std
fig, axes = plt.subplots(5, 4)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, nfy, nfx)
    indC = codeLst.index(code)
    indS = np.where(count[:, indC] > the)[0]
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {} {}'.format(code, shortName, len(indS))
    # a = np.log(mean[indS, indC]+1e-5)
    # b = np.log(std[indS, indC]+1e-5)
    a = mean[indS, indC]
    b = std[indS, indC]
    axplot.plot121(axes[j, i], a, b)    
    axplot.titleInner(axes[j, i], titleStr)
fig.show()

# examine for extremes
code = '00665'
indC = DF.varC.index(code)
# a = np.nanmax(out[:, :, indC], axis=0)
a = np.log(std[:, indC]+1e-5)
b = count[:, indC]
indS = np.where((a < -6) & (b > 100))[0]
k = 0
fig, axes = plt.subplots(2, 1)
axplot.plotTS(axes[0], DF.t, DF.c[:, indS[k], indC])
axplot.plotTS(axes[1], DF.t, out[:, indS[k], indC])
print(mean[indS[0], indC], std[indS[k], indC])
fig.show()
