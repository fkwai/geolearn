from hydroDL import kPath,utils
from hydroDL.app import waterQuality, DGSA
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnS', trainSet)
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
corrMat = np.full([len(siteNoLst),  2], np.nan)
rmseMat = np.full([len(siteNoLst),  2], np.nan)
ic = wqData.varC.index(code)
for iT, subset in enumerate([trainSet, testSet]):
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData, ep=ep, reTest=reTest)
    ind = wqData.subset[subset]
    info = wqData.info.iloc[ind].reset_index()
    o = wqData.c[-1, ind, ic]
    p = yP[-1, :, 1]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iT] = corr
        rmseMat[iS, iT] = rmse

dfG = gageII.readData(varLst=gageII.varLst, siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)

pMat = dfG.values
dfS = DGSA.DGSA_light(
    pMat, corrMat, ParametersNames=dfG.columns.tolist(), n_clsters=5)
# ax = dfS.sort_values(by=0).plot.barh()
# plt.show()

dfSP = dfS.sort_values(by=0)
fig, ax = plt.subplots(1, 1)
x = range(len(dfSP))
cLst = list()
for b in (dfSP[0] > 1).tolist():
    cLst.append('r') if b is True else cLst.append('b')
ax.barh(x, dfSP[0].values, color=cLst)
ax.set_yticks(x)
ax.set_yticklabels(dfSP.index.tolist())
plt.tight_layout()
fig.show()
