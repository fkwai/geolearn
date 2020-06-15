from astropy.timeseries import LombScargle
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

import scipy.signal as signal

wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst

# real data
indS = np.random.randint(0, 64)
siteNo = siteNoLst[indS]

dfC = waterQuality.readSiteY(siteNo, ['00955'])
startDate = np.datetime64('1979-01-01')
dfQ = usgs.readStreamflow(siteNo)
dfQ = dfQ[dfQ.index > startDate]
dfF = gridMET.readBasin(siteNo)
dfF = dfF[dfF.index > startDate]
# rm outlier
df = dfC[dfC['00955'].notna().values]
y = df['00955'].values
yV = y[y < np.percentile(y, 99)]
yV = yV[yV > np.percentile(y, 1)]
ul = np.mean(yV)+np.std(yV)*5
dfC[dfC['00955'] > ul] = np.nan
dfC = dfC['00955']
dfQ = dfQ['00060_00003']
dfF = dfF['pr']

# fourier
fig, axes = plt.subplots(3, 1, figsize=(10, 4))
for k, dfObs in enumerate([dfC, dfQ, dfF]):
#     dfObs = dfQ
    print(k)
    df = dfObs[dfObs.notna().values]
    tt = dfObs.index.values
    xx = (tt.astype('datetime64[D]') -
          np.datetime64('1979-01-01')).astype(np.float)
    t = df.index.values
    x = (t.astype('datetime64[D]') -
         np.datetime64('1979-01-01')).astype(np.float)
    y = df.values
    y = y-np.nanmean(y)
    nt = len(xx)
    freq = np.fft.fftfreq(nt)[1:]

    ls = LombScargle(x, y)
    power = ls.power(freq)
    xx = (dfObs.index.values.astype('datetime64[D]') -
          np.datetime64('1979-01-01')).astype(np.float)

    p = ls.false_alarm_probability(power)

    ind = np.where(p < 0.2)[0]
    pd = np.unique(np.abs((1/freq[ind]).astype(int)))
    ind = np.where(1/freq > 0)[0]
    ind = np.where((1/freq >= 0) & (1/freq < 1000))[0]
    # axes[0].plot(1/freq[ind], 1-p[ind], '-r', label='baluev probability')
    axes[k].plot(1/freq[ind], power[ind], '-*b')
    axes[k].legend()
#     axes[k].set_xlabel('period (day)')
plt.tight_layout()
fig.show()
