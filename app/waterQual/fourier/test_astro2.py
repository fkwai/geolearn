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

# wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst

# real data
# siteNo = '07083000'
# siteNo = '09352900'
indS = np.random.randint(0, 64)
siteNo = siteNoLst[indS]
# siteNo = '06317000'
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
# nt = 1000
# freq = 1/np.linspace(2, nt, nt)
# freq = np.arange(1, nt)/nt
freq = np.fft.fftfreq(nt)[1:]

yy = dfObs['00955'].values
np.fft.fft(yy)

ls = LombScargle(x, y)
power = ls.power(freq)
xx = (dfObs.index.values.astype('datetime64[D]') -
      np.datetime64('1979-01-01')).astype(np.float)

p = ls.false_alarm_probability(power)

indP = np.where(p < 0.1)[0]
pd = np.unique(np.abs((1/freq[indP]).astype(int)))

yy = np.zeros(len(tt))
y2 = np.zeros(len(t))
# for d in pd:
#     if d > 0:
#         yy = yy+ls.model(xx, 1/d)
#         y2 = y2+ls.model(x, 1/d)
for k in indP.tolist():
      yy = yy+ls.model(xx, freq[k])
      y2 = y2+ls.model(x, freq[k])

fig, axes = plt.subplots(2, 1, figsize=(10, 4))
axes[0].plot(tt, yy, '-r', label='Lomb-Scargle')
axes[0].plot(t, y, '-*b', label='obs')
axes[0].legend()
# axes[0].set_xlabel('day')
axes[0].set_title(siteNo)
axes[1].plot(t, y2-y, '-*b', label='obs')
axes[1].legend()
axes[1].set_xlabel('period (day)')
plt.tight_layout()
fig.show()
