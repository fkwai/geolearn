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
nS = 200
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]
dfG = dfGeo.loc[siteNoLst]




varGLst = dfG.columns.tolist()
dfRsq = pd.DataFrame(index=varGLst, columns=['Rsq1', 'Rsq2'])
for varG in varGLst:
    x = dfG[varG].values
    y1 = dfR1['corr'].values
    y2 = dfR1['corr'].values
    (xx, yy1, yy2), _ = utils.rmNan([x, y1, y2])
    r1 = np.corrcoef(xx, yy1)[0, 1]
    dfRsq.at[varG, 'Rsq1'] = r1**2
    r2 = np.corrcoef(xx, yy2)[0, 1]
    dfRsq.at[varG, 'Rsq2'] = r2**2

dfRsq.to_csv('temp')
dfRsq.sort_values('Rsq1', ascending=False)

# varG = 'SLOPE_PCT'
varG = 'HLR_BAS_PCT_100M'
x = dfG[varG].values
y = dfR1['corr'].values
x[x < -900] = np.nan
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, '*')
ax.set_xlabel(varG)
ax.set_ylabel('WRTDS corr')
fig.show()

# map
lat = dfG['LAT_GAGE'].values
lon = dfG['LNG_GAGE'].values
figM, axM = plt.subplots(1, 2, figsize=(12, 4))
axplot.mapPoint(axM[0], lat, lon, x, s=16)
axplot.mapPoint(axM[1], lat, lon, y, vRange=[0, 1], s=16)
shortName = usgs.codePdf.loc[code]['shortName']
axM[0].set_title(varG)
axM[1].set_title('WRTDS corr, {}'.format(shortName))
figM.show()
