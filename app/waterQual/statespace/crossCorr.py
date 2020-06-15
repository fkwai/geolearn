import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst
varX = ['00060', 'pr']
nt = 365*3
ns = len(siteNoLst)
nx = len(varX)
corrMat = np.zeros([ns, nt, nx])
for iS in range(ns):
    iS
    siteNo = siteNoLst[iS]
    dfY = waterQuality.readSiteY(siteNo, ['00955'])
    dfY = dfY.dropna()
    dfX = waterQuality.readSiteX(siteNo, varX)
    t = dfY.index
    y = dfY['00955'].values
    for k in range(nt):
        x = dfX.loc[t.values-np.timedelta64(k, 'D')].values
        ind = np.where(~np.isnan(x))[0]
        for i in range(nx):
            corrMat[iS, k, i] = np.corrcoef(x[ind, i], y[ind])[0, 1]

indS = np.random.randint(0, ns)
fig, ax = plt.subplots(1, 1)
ax.plot(np.arange(nt), corrMat[indS, :, 0].T, '-*')
fig.show()


# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(8, 6))
    axplot.mapPoint(axM, lat, lon, corrMat[:, 0, 0], s=12)
    figP, axP = plt.subplots(1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    axP.plot(np.arange(nt), corrMat[iP, :, 0].T, '-*')


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)
