from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from scipy import stats


# load WRTDS results
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')

# dfRes1[dfRes1 == -9999] = np.nan
dfGeo = gageII.readData(siteNoLst=dfRes1.index.tolist())
dfGeo = gageII.updateCode(dfGeo)

# select sites
nS = 100
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]
dfG = dfGeo.loc[siteNoLst]


# hist
fig, ax = plt.subplots(1, 1)
ax.hist(dfR1['corr'], bins=20)
fig.show()

# cdf
x = dfR1['corr'].values
# x = np.random.normal(0, 1, size=100)
xr, ind = utils.rankData(x)
yr = np.arange(len(x))/len(x)
fig, ax = plt.subplots(1, 1)
ax.plot(xr, yr, '-k', label='correlations')
# distLst = ['norm', 'beta']
distLst = ['norm']
colorLst = 'rgbmcy'
for distName, color in zip(distLst, colorLst):
    dist = getattr(stats, distName)
    arg = dist.fit(xr)
    cdf = dist.cdf(xr, *arg)
    label = ' '.join([distName]+['{:.2f}'.format(x) for x in arg])
    ax.plot(xr, cdf, color, label=label)
ax.legend()
fig.show()


# attributes
varG = 'CLAYAVE'
x = dfG[varG].values
# x = np.random.rand(len(x))
y = dfR1['corr'].values
x[x < -900] = np.nan

# density plot
vLst = np.arange(0, 1, 0.1)
dataBox = list()
labLst = list()
for k in range(1, len(vLst)):
    v1 = vLst[k-1]
    v2 = vLst[k]
    ind = np.where((y >= v1) & (y < v2))[0]
    if len(ind) > 0:
        dataBox.append(x[ind])
        labLst.append('{:.2f}'.format(v1))
vRange = [np.nanmin(x), np.nanmax(x)]
fig = figplot.boxPlot(dataBox, label1=labLst, figsize=(8, 4), widths=0.3)
plt.subplots_adjust(wspace=0)
fig.show()

# cum plot
yr, ind = utils.rankData(y)
yr = yr[::-1]
ind = ind[::-1]
xr = np.cumsum(x[ind])/np.arange(1, len(ind)+1)
fig, ax = plt.subplots(1, 1)
ax.plot(yr[10:], xr[10:], '-')
fig.show()

# 121
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, '*')
ax.set_xlabel(varG)
ax.set_ylabel('WRTDS corr')
fig.show()
