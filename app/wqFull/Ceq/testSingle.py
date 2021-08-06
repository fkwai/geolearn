
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time
import statsmodels.api as sm
from hydroDL.data import dbBasin, gageII, usgs
from hydroDL.master import basinFull

# calculate Ceq

dataName = 'G200'
# DF = dbBasin.DataFrameBasin(dataName)

sn = 1e-5
thP = 0.0001
thR = 0.5

code = '00915'
siteNo = '06800000'
siteNo = DF.siteNoLst[10]

indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
Q = DF.q[:, indS, 1]
C = DF.c[:, indS, indC]
q = np.log(Q+sn)
c = np.log(C+sn)

#  made up data
# q = np.linspace(0, 1, 100)
# c = np.ndarray(100)
# kk = 10
# s = 5
# c[:kk] = np.random.random(kk)
# c[kk:] = np.random.random(100-kk)+q[kk:]*-s-q[kk]*-s

[x, y], _ = utils.rmNan([q, c])
ind = np.argsort(x)
n = len(y)
pLst = list()
bLst = list()
rLst = list()
aLst = list()
for k in range(n):
    xx = x[ind[k:]]
    yy = y[ind[k:]]
    mod = sm.OLS(yy, sm.add_constant(xx))
    res = mod.fit()
    p = res.pvalues[1]
    a = res.params[0]
    b = res.params[1]
    r = res.rsquared
    rLst.append(r)
    pLst.append(p)
    bLst.append(b)
    aLst.append(a)

# plot
# k = np.where(np.array(pLst) < thP)[0][0]
k = np.where(np.array(rLst) > thR)[0][0]
fig, ax = plt.subplots(1, 1)
x1 = x[ind[k]]
x2 = np.nanmax(x)
y1 = x1*bLst[k]+aLst[k]
y2 = x2*bLst[k]+aLst[k]
ax.plot(x, y, 'k*')
ax.plot(x[ind[k]], y[ind[k]], 'r.')
ax.plot([x1, x2], [y1, y2], 'r-')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(np.log10(pLst))
ax.twinx().plot(bLst, 'r')
fig.show()

data = np.log10(pLst[:-1])
np.corrcoef(data[:-1], data[1:])[0]

# timeseries
t = DF.t
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(t, Q, '-b')
ax.twinx().plot(t, C, '*r')
ax.xaxis_date()
fig.show()
