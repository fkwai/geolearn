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
freq = 'W'
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')

# select sites
nS = 200
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]


#
k = 10
siteNo = siteNoLst[k]
dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
x = dfO.dropna().values.flatten()
# cdf
xr, ind = utils.rankData(x)
yr = np.arange(len(x))/len(x)
fig, ax = plt.subplots(1, 1)
ax.plot(xr, yr, '-k', label='correlations')
distLst = ['norm', 'beta']
# distLst = ['norm']
colorLst = 'rgbmcy'
for distName, color in zip(distLst, colorLst):
    dist = getattr(stats, distName)
    arg = dist.fit(xr)
    cdf = dist.cdf(xr, *arg)
    s, p = stats.kstest(x, distName, arg)
    # label = ' '.join([distName]+['{:.2f}'.format(x) for x in arg])
    label = '{} {:.3f}'.format(distName, p)
    ax.plot(xr, cdf, color+'*-', label=label)
ax.legend()
fig.show()

x = np.random.normal(0, 1, size=100)
stats.normaltest(x)
stats.shapiro(x)


distName = 'beta'
dist = getattr(stats, distName)
arg = dist.fit(x)
stats.kstest(x, distName, arg)
arg = dist.fit(xr)
stats.kstest(xr, distName, arg)

stats.shapiro(x)