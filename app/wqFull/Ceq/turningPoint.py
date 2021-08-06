
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
thP = 0.0001
thR = 0.6

code = '00915'
[matK, matN, matA, matB, matP, matR] = [
    np.full(len(DF.siteNoLst), np.nan) for x in range(6)]
# siteNo = '06800000'
matR = np.full(len(DF.siteNoLst), np.nan)
matP = np.full(len(DF.siteNoLst), np.nan)
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
    if n > 20:
        pLst = list()
        for k in range(n-1):
            xx = x[ind[k:]]
            yy = y[ind[k:]]
            mod = sm.OLS(yy, sm.add_constant(xx))
            res = mod.fit()
            p = res.pvalues[1]
            a = res.params[0]
            b = res.params[1]
            r = res.rsquared
            pLst.append(p)
        pAry = np.log(np.array(pLst))
        matR[indS] = np.nanargmin(pAry)/len(pAry)
        matP[indS] = np.nanmin(pAry)

fig, ax = plt.subplots(1, 1)
ax.plot(pAry)
fig.show()

temp = matR
code = '00915'
indC = DF.varC.index(code)
thP = -100
ind1 = np.where((matR <= 0.2) & (matP < thP))[0]
ind2 = np.where((matR <= 0.2) & (matP > thP))[0]
ind3 = np.where(matR > 0.2)[0]
fig, axes = plt.subplots(3, 1)
for ax, ind in zip(axes, [ind1, ind2, ind3]):
    axplot.plot121(ax, matL[ind, indC, 3], matW[ind, indC, 3])
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(matR, matP, '*')
fig.show()
