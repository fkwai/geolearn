from hydroDL import kPath
from hydroDL.app import waterQuality, DGSA
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib
from astropy.timeseries import LombScargle


import pandas as pd
import numpy as np
import os
import time

wqData = waterQuality.DataModelWQ('Silica64')

siteNoLst = wqData.siteNoLst
dictP = dict()
for k, siteNo in enumerate(siteNoLst):
    print(siteNo)
    dfObs = waterQuality.readSiteY(siteNo, ['00955'])
    dfObs = waterQuality.readSiteY(siteNo, ['00955'])
    # rm outlier
    df = dfObs[dfObs['00955'].notna().values]
    y = df['00955'].values
    yV = y[y < np.percentile(y, 99)]
    yV = yV[yV > np.percentile(y, 1)]
    ul = np.mean(yV)+np.std(yV)*5
    dfObs[dfObs['00955'] > ul] = np.nan
    # fourier
    df = dfObs[dfObs.notna().values]
    tt = dfObs.index.values
    xx = (tt.astype('datetime64[D]') -
          np.datetime64('1979-01-01')).astype(np.float)
    t = df.index.values
    x = (t.astype('datetime64[D]') -
         np.datetime64('1979-01-01')).astype(np.float)
    y = df['00955'].values
    y = y-np.nanmean(y)
    nt = len(xx)
    freq = np.fft.fftfreq(nt)[1:]
    ls = LombScargle(x, y)
    power = ls.power(freq)
    prob = ls.false_alarm_probability(power)

    ind = np.where(prob < 0.05)[0]
    pd = np.unique(np.abs((1/freq[ind]).astype(int)))
    dictP[siteNo] = pd.tolist()

pLst = sum(list(dictP.values()), [])
pu, pc = np.unique(np.array(pLst), return_counts=True)
temp = np.stack([pu, pc]).transpose()

rMat = np.zeros([len(siteNoLst), 3])
for k, siteNo in enumerate(siteNoLst):
    temp = dictP[siteNo]
    if 6 in temp or 7 in temp:
        rMat[k, 0] = 1
    if 182 in temp:
        rMat[k, 1] = 1
    if 365 in temp:
        rMat[k, 2] = 1


# plot map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
figM, axM = plt.subplots(3, 1, figsize=(8, 6))
for k in range(3):
    mm = axplot.mapPoint(axM[k], lat, lon, rMat[:, k], s=12)
axM[0].set_title('weekly signal')
axM[1].set_title('half yearly signal')
axM[2].set_title('yearly signal')
figM.show()

dfG = gageII.readData(varLst=gageII.varLst, siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)

pMat = dfG.values
dfS = DGSA.DGSA_light(
    pMat, rMat, ParametersNames=dfG.columns.tolist(), n_clsters=3)
ax = dfS.sort_values(by=0).plot.barh()
plt.show()

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
