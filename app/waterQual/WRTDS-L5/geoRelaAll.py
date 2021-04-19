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

# attributes
attrLst = dfG.columns.tolist()
dfOut = pd.DataFrame(index=attrLst, columns=['dist'])
y = dfR1['corr'].values
yr, ind = utils.rankData(y)
yr = yr[::-1]
ind = ind[::-1]
fig, ax = plt.subplots(1, 1)
for varG in attrLst:
    x = dfG[varG].values
    # x = np.random.normal(0, 1, size=len(x))
    temp = x.argsort()
    xr = np.arange(len(x))[temp.argsort()]/len(x)
    # cum plot
    xc = np.cumsum(xr[ind])/np.arange(1, len(ind)+1)
    ax.plot(yr[10:], xc[10:], 'k-', alpha=0.2)
    dfOut.at[varG, 'dist'] = np.abs(np.nanmean(xc-0.5))
fig.show()
dfOut.to_csv('temp')


# 121
varG='WDMAX_SITE'
x = dfG[varG].values
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, '*')
ax.set_xlabel(varG)
ax.set_ylabel('WRTDS corr')
fig.show()
