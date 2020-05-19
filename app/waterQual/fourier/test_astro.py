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
# siteNo = '07083000'
siteNo = '09352900'
indS = np.random.randint(0, 64)
siteNo = siteNoLst[indS]
# siteNo = '04193500'
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

ls = LombScargle(x, y)
power = ls.power(freq)
xx = (dfObs.index.values.astype('datetime64[D]') -
      np.datetime64('1979-01-01')).astype(np.float)

p = ls.false_alarm_probability(power)
# ym = np.zeros([len(freq), len(x)])
# yp = np.zeros([len(freq), len(x)])
# for k, f in enumerate(freq):
#     ym[k, :] = ls.model(x, f)
#     yp[k, :] = ym[k, :]*ls.power(f)
folder = r'C:\Users\geofk\work\waterQuality\tempData\LS'
ym = np.load(os.path.join(folder, siteNo)+'.npy')
yp = np.load(os.path.join(folder, siteNo+'-full')+'.npy')


ind = np.where(p < 0.2)[0]
pd = np.unique(np.abs((1/freq[ind]).astype(int)))

fig, axes = plt.subplots(2, 1, figsize=(10, 4))
axes[0].plot(t, y, '-*b', label='obs')
axes[0].plot(t, np.mean(ym, axis=0)/2*len(t), '-r', label='Lomb-Scargle')
axes[0].legend()
# axes[0].set_xlabel('day')
axes[0].set_title(siteNo)
ind = np.where(1/freq > 0)[0]
ind = np.where((1/freq >= 0) & (1/freq < 1000))[0]
axes[1].plot(1/freq[ind], 1-p[ind], '-r', label='baluev probability')
axes[1].plot(1/freq[ind], power[ind], '-*b', label='power')
# axes[1].plot(1/freq, power)
# axes[1].set_ylabel('power')
axes[1].legend()
axes[1].set_xlabel('period (day)')
plt.tight_layout()
fig.show()


prLst = [365*40, 1000, 300, 150, 60, 10, 0]
frLst = [0]+[1/x for x in prLst[1:-1]]+[1]
nf = len(frLst)-1
fig, axes = plt.subplots(nf, 1, figsize=(10, 6))
for kf in range(nf):
    freqAbs = np.abs(freq)
    ind = np.where((freqAbs >= frLst[kf]) & (freqAbs < frLst[kf+1]))[0]
    axes[kf].plot(tt, np.sum(yp[ind, :]/2*len(t)/len(tt),
                             axis=0), '-r')
    axes[kf].set_title('T range [{},{}]'.format(prLst[kf], prLst[kf+1]))
    # axes[kf].xaxis.set_ticklabels([])
plt.tight_layout()
fig.show()


# load forcing
startDate = np.datetime64('1979-01-01')
dfQ = usgs.readStreamflow(siteNo)
dfQ = dfQ[dfQ.index > startDate]
dfF = gridMET.readBasin(siteNo)
dfF = dfF[dfF.index > startDate]

fig, axes = plt.subplots(3, 1, figsize=(10, 6))
freqAbs = np.abs(freq)
ind = np.where((freqAbs >= 1/10) & (freqAbs < 1.1))[0]
axes[0].plot(t, y, '-*b', label='obs')
axes[0].plot(tt, np.sum(yp[ind, :]/2*len(t)/len(tt), axis=0), '-r', label='high freq')
axes[0].legend()
axT1 = axes[1].twinx()
axT1.plot(dfQ, '-b', label='streamflow')
axes[1].plot(tt, np.sum(yp[ind, :]/2*len(t)/len(tt), axis=0), '-r', label='high freq')
axes[1].legend()
axT1.legend()
axT2 = axes[2].twinx()
axT2.plot(dfF['pr'], '-b', label='prcp')
axes[2].plot(tt, np.sum(yp[ind, :]/2*len(t)/len(tt), axis=0), '-r', label='high freq')
axes[2].legend()
axT2.legend()
plt.tight_layout()
fig.show()

