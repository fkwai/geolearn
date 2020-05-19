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

mapMat = np.zeros([ns, nx])
for iS in range(ns):
    siteNo = siteNoLst[iS]
    dfY = waterQuality.readSiteY(siteNo, ['00955'])
    dfY = dfY.dropna()
    dfX = waterQuality.readSiteX(siteNo, varX)
    t = dfY.index
    y = dfY['00955'].values
    x = dfX.loc[t.values].values
    ind = np.where(~np.isnan(x))[0]
    for i in range(nx):
        mapMat[iS, i] = np.corrcoef(x[ind, i], y[ind])[0, 1]

# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(8, 6))
    axplot.mapPoint(axM, lat, lon, mapMat[:, 0], s=12)
    figP, axP = plt.subplots(3, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfY = waterQuality.readSiteY(siteNo, ['00955'])
    dfY = dfY.dropna()
    dfX = waterQuality.readSiteX(siteNo, varX)
    t = dfY.index
    y = dfY['00955'].values
    corrMat = np.zeros([nt, nx])
    for k in range(nt):
        x = dfX.loc[t.values-np.timedelta64(k, 'D')].values
        ind = np.where(~np.isnan(x))[0]
        for i in range(nx):
            corrMat[k, i] = np.corrcoef(x[ind, i], y[ind])[0, 1]
    axP[0].plot(dfX['00060'], '-b', label='streamflow')
    axP[1].plot(dfY, '-*r', label='silica')
    axP[2].plot(np.arange(nt), corrMat[:, 0].T, '-*')
    axP[2].set_ylabel('correlation')
    axP[2].set_xlabel('lag day')



importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)


figP, axP = plt.subplots(2, 1, figsize=(8, 6))
iP = 0
siteNo = siteNoLst[iP]
dfY = waterQuality.readSiteY(siteNo, ['00955'])
dfY = dfY.dropna()
dfX = waterQuality.readSiteX(siteNo, varX)
t = dfY.index
y = dfY['00955'].values
corrMat = np.zeros([nt, nx])
for k in range(nt):
    x = dfX.loc[t.values-np.timedelta64(k, 'D')].values
    ind = np.where(~np.isnan(x))[0]
    for i in range(nx):
        corrMat[k, i] = np.corrcoef(x[ind, i], y[ind])[0, 1]
axP[0].plot(dfX['00060'])
axP[1].plot(np.arange(nt), corrMat[:, 0].T, '-*')
figP.show()
