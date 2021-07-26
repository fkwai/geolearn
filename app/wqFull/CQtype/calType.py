
from mpl_toolkits import basemap
import pandas as pd
from torch._C import is_autocast_enabled
from hydroDL.data import dbBasin, gageII
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

# only selected sites
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictG200.json')) as f:
    dictSite = json.load(f)

dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

siteNoLst = DF.siteNoLst
codeLst = DF.varC
ns = len(siteNoLst)
nc = len(codeLst)
matA = np.full([ns, nc, 2], np.nan)
matB = np.full([ns, nc, 2], np.nan)
matP = np.full([ns, nc, 2], np.nan)
t0 = time.time()
for indS, siteNo in enumerate(siteNoLst):
    for indC, code in enumerate(codeLst):
        Q = DF.q[:, indS, 1]
        C = DF.c[:, indS, indC]
        nobs = np.sum(~np.isnan(Q) & ~np.isnan(C))
        if nobs > 50:
            a, b, p = cqType.calPar(Q, C)
            matA[indS, indC, :] = a
            matB[indS, indC, :] = b
            matP[indS, indC, :] = p
    t1 = time.time()
    print('{} {} {:.2f}'.format(indS, siteNo, t1-t0))

dirSave = os.path.join(kPath.dirWQ, 'modelStat', 'typeCQ', dataName)
np.savez(dirSave, matA=matA, matB=matB, matP=matP)

code = '00955'
indC = codeLst.index(code)
th = 0.05
h1 = (matP[:, indC, 0] < th).astype(int)
s1 = (matB[:, indC, 0] > 0).astype(int)+1
h2 = (matP[:, indC, 1] < th).astype(int)
s2 = (matB[:, indC, 1] > 0).astype(int)+1
tp = h1*s1*3+h2*s2
tp[np.any(np.isnan(matP[:, indC, :]), axis=1)] = -1
v, count = np.unique(tp, return_counts=True)


importlib.reload(axplot)
importlib.reload(cqType)
tp = cqType.par2type(matB, matP)

# plot map
code = '00955'
indC = codeLst.index(code)
tpC = tp[:, indC]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
fig, ax = plt.subplots(1, 1)
vLst, cLst,  mLst, labLst = cqType.getPlotArg()
axplot.mapPointClass(ax, lat, lon, tp[:, indC], vLst=vLst, mLst=mLst,
                     cLst=cLst, labLst=labLst)
fig.show()

# CQ plot
indS = np.where(tpC == 4)[0][10]
fig, ax = plt.subplots(1, 1)
Q = DF.q[:, indS, 1]
C = DF.c[:, indS, indC]
a = matA[indS, indC, :]
b = matB[indS, indC, :]
p = matP[indS, indC, :]
cqType.plotCQ(ax, Q, C, a, b, p)
fig.show()
