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
import statsmodels.api as sm
import time

DF = dbBasin.DataFrameBasin('G400Norm')
trainSet = 'rmRT20'
testSet = 'all'

h = [7, 2, 0.5]
the = 100
sn = 1e-5

# Calculate WRTDS from train and test set
varX = ['00060', 'sinT', 'cosT', 'datenum']
varY = usgs.newC
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varX=varX, varY=varY)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varX=varX, varY=varY)
tt = pd.to_datetime(DF.t)
yr = tt.year.values
t = yr+tt.dayofyear.values/365
###
yOut = np.full([len(d2.t), len(d2.siteNoLst), len(varY)], np.nan)
t0 = time.time()
# for indS, siteNo in enumerate(d2.siteNoLst):
siteNo = '11074000'
indS = d2.siteNoLst.index(siteNo)

for indC, code in enumerate(varY):
    print('{} {} {} {}'.format(indS, siteNo, code, time.time()-t0))
    x1 = d1.X[:, indS, :].copy()
    y1 = d1.Y[:, indS, indC].copy()
    q1 = x1[:, 0].copy()
    q1[q1 < 0] = 0
    logq1 = np.log(q1+sn)
    x1[:, 0] = logq1
    x2 = d2.X[:, indS, :].copy()
    y2 = d2.Y[:, indS, indC].copy()
    q2 = x2[:, 0].copy()
    q2[q2 < 0] = 0
    logq2 = np.log(q2+sn)
    x2[:, 0] = logq2
    [xx1, yy1], ind1 = utils.rmNan([x1, y1])
    if testSet == 'all':
        [xx2], ind2 = utils.rmNan([x2])
    else:
        [xx2, yy2], ind2 = utils.rmNan([x2, y2])
    if len(ind1) == 0:
        continue
    for k in ind2:
        dY = np.abs(t[k]-t[ind1])
        dQ = np.abs(logq2[k]-logq1[ind1])
        dS = np.min(
            np.stack([abs(np.ceil(dY)-dY), abs(dY-np.floor(dY))]), axis=0)
        d = np.stack([dY, dQ, dS])
        n = d.shape[1]
        if n > the:
            hh = np.tile(h, [n, 1]).T
            bW = False
            while ~bW:
                bW = np.sum(np.all(hh-d > 0, axis=0)) > the
                hh = hh*1.1 if not bW else hh
        else:
            htemp = np.max(d, axis=1)*1.1
            hh = np.repeat(htemp[:, None], n, axis=1)
        w = (1-(d/hh)**3)**3
        w[w < 0] = 0
        wAll = w[0]*w[1]*w[2]
        ind = np.where(wAll > 0)[0]
        ww = wAll[ind]

        model = sm.WLS(yy1[ind], xx1[ind], weights=ww).fit()
        yp = model.predict(x2[k, :])[0]
        yOut[k, indS, indC] = yp

code = '00915'
fig, ax = plt.subplots(1, 1)
indC = varY.index(code)
a = yOut[:, indS, indC]
b = d1.Y[:, indS, indC]
c = d2.Y[:, indS, indC]
axplot.plotTS(ax, DF.t, [c, b, a])
fig.show()

fig, ax = plt.subplots(1, 1)
indC = varY.index(code)
a = x1[:, 0]
b = x2[:, 0]
axplot.plotTS(ax, DF.t, [b, a])
fig.show()
