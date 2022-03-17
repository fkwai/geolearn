
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

# load models
dataName = 'G200N'
DFN = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.newC
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(
    outName, DF=DFN, testSet=testSet, ep=500)
yL = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DFN.g[:, DFN.varG.index(code+'-M')]
    s = DFN.g[:, DFN.varG.index(code+'-S')]
    yL[:, :, k] = yP[:, :, k]*s+m
siteNoLst = DFN.siteNoLst
ns = len(siteNoLst)
nc = len(codeLst)

# load WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# correlation matrix
d1 = dbBasin.DataModelBasin(DFN, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DFN, subset=testSet, varY=codeLst)
siteNoLst = DFN.siteNoLst
matW = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
matL = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            statL = utils.stat.calStat(yL[:, indS, indC], d2.Y[:, indS, indC])
            matL[indS, indC, :] = list(statL.values())


dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

sn = 1e-5
thP = 0.001
thR = 0.5

code = '00915'
[matK, matN, matA, matB, matP, matR] = [
    np.full(len(DF.siteNoLst), np.nan) for x in range(6)]
# siteNo = '06800000'

for indS, siteNo in enumerate(DF.siteNoLst):
    indS = DF.siteNoLst.index(siteNo)
    indC = DF.varC.index(code)
    Q = DF.q[:, indS, 1]
    C = DF.c[:, indS, indC]
    q = np.log(Q+sn)
    c = np.log(C+sn)
    [x, y], _ = utils.rmNan([q, c])
    ind = np.argsort(x)
    n = len(y)
    if n > 10:
        for k in range(n):
            xx = x[ind[k:]]
            yy = y[ind[k:]]
            mod = sm.OLS(yy, sm.add_constant(xx))
            res = mod.fit()
            p = res.pvalues[1]
            a = res.params[0]
            b = res.params[1]
            r = res.rsquared
            # print(r)
            # if p > thP:
            if r < thR:
                continue
            else:
                break
        matK[indS] = k
        matN[indS] = n-1
        matA[indS] = a
        matB[indS] = b
        matP[indS] = p
        matR[indS] = r

# plot
fig, ax = plt.subplots(1, 1)
x1 = x[ind[k]]
x2 = np.nanmax(x)
y1 = x1*b+a
y2 = x2*b+a
ax.plot(x, y, 'k*')
ax.plot(x[ind[k]], y[ind[k]], 'r.')
ax.plot([x1, x2], [y1, y2], 'r-')
fig.show()

# fig, ax = plt.subplots(1, 1)
# cs = ax.plot(x, y, 'k-', alpha=0.3)
# cs = ax.scatter(x, y, c=m)
# month = DF.t.astype('datetime64[M]').astype(int) % 12 + 1
# fig.show()


temp = matK/matN
# temp[temp > 0.8] = 1
code = '00915'
indC = DF.varC.index(code)


fig, ax = plt.subplots(1, 1)
axplot.scatter121(ax, matL[:, indC, 3], matW[:, indC, 3], temp)
fig.show()
