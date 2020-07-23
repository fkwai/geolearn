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


siteNo = '01674500'
code = '00955'
outName = 'Silica64-Y8090-00955-opt1'

dfP1, dfObs = basins.loadSeq(outName, siteNo)

dfP2 = wqLinear.loadSeq(siteNo, code, 'ARMA', optT='Y8090', order=(5, 0, 0))
rmse2, corr2 = waterQuality.calErrSeq(dfP2[code], dfObs[code])
t = dfObs.index.values

tBar = np.datetime64('2000-01-01')
styLst = '-*'
figP, axP = plt.subplots(1, 1, figsize=(8, 6))

axplot.plotTS(axP, t, [dfP2[code], dfObs[code]],
              tBar=tBar, styLst='-*', cLst='br')
figP.show()


dfO = dfObs[code]
dfP = dfP2[code]
sd = np.datetime64('1980-01-01')
a = dfO[dfO.index >= tBar].values
b = dfP[dfP.index >= tBar].values
t=dfO[dfO.index >= tBar].index.values
indV = np.where(~np.isnan(a))
rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
corr = np.corrcoef(a[indV], b[indV])[0, 1]

figP, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(a, b, '*')
figP.show()

figP, axP = plt.subplots(1, 1, figsize=(8, 6))
# axplot.plotTS(axP, t, [a, b],tBar=tBar, styLst='**', cLst='br')
axplot.plotTS(axP, t[indV], [a[indV], b[indV]],tBar=tBar, styLst='**', cLst='br')
figP.show()

