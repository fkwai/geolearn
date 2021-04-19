
import importlib
import numpy as np
import os
import pandas as pd
import json
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import usgs, gageII
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
codeLst = sorted(usgs.newC)

# load Linear and Seasonal model
dictL = dict()
dirL = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DL', 'B10', 'output')
dictS = dict()
dirS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DS', 'B10', 'output')
for dirTemp, dictTemp in zip([dirL, dirS], [dictL, dictS]):
    for k, siteNo in enumerate(siteNoLst):
        print('\t WRTDS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirTemp, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        dictTemp[siteNo] = df

dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t USGS site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(
        siteNo, varLst=['00060']+codeLst, freq='W', rmFlag=True)
    dictObs[siteNo] = df

# calculate rsq
rMat = np.full([len(siteNoLst), len(codeLst), 2], np.nan)
tt = np.datetime64('2010-01-01')
t0 = np.datetime64('1980-01-01')
t = dictObs[siteNoLst[0]].index.values
ind1 = np.where((t < tt) & (t >= t0))[0]
ind2 = np.where(t >= tt)[0]
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictL[siteNo][code].values
        v2 = dictS[siteNo][code].values
        v0 = dictObs[siteNo][code].values
        (vv0, vv1, vv2), indV = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(vv1, vv0)
        rmse2, corr2 = utils.stat.calErr(vv2, vv0)
        rMat[indS, ic, 0] = corr1**2  # linearity
        rMat[indS, ic, 1] = corr2**2  # seasonality

# a cdf for rsq of seasonality and linearity
code = '00915'
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
ic = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
axplot.plotCDF(ax, [rMat[indS, ic, 0], rMat[indS, ic, 1]],
               legLst=['linearity', 'seasonality'])
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(rMat[indS, ic, 0], rMat[indS, ic, 1], 'r*')
fig.show()


len(np.where(rMat[indS, ic, 0] > 0.5)[0])/len(indS)
