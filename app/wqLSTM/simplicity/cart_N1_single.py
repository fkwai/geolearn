

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import matplotlib.gridspec as gridspec

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)

matRm = count < 200

# load linear/seasonal
dirP = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'
labLst = ['Q', 'S', 'QS']
dictS = dict()
for lab in labLst:
    dirS = dirP.format(lab)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirS, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictS[lab] = matLR

matQ = dictS['Q']
matS = dictS['S']
matQS = dictS['QS']
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
lat, lon = DF.getGeo()

code = '00915'
ic = codeLst.index(code)
[mat], indS = utils.rmNan([matQS[:, ic]])
siteNoCode = [DF.siteNoLst[ind] for ind in indS]
dfGC = dfG.loc[siteNoCode]

aLst = ['NO10AVE', 'NO4AVE']
vLst = [78.27499771, 82.71999741]

a = aLst[0]
v = vLst[0]
x = dfGC[a]
ind1 = np.where(x.values <= v)
ind2 = np.where(x.values > v)

fig, ax = plt.subplots(1, 1)
ax.plot(x, mat, '*')
codeStr = usgs.codePdf.loc[code]['shortName']
ax.set_title('{} {}'.format(code, codeStr))
ax.set_xlabel(a)
ax.set_ylabel('simplicity')
fig.show()

fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(2, 1)
ax1 = mapplot.mapPoint(fig, gs[0, 0], lat[indS], lon[indS], mat, cb=True)
ax1.set_title('simplicity')
ax2 = mapplot.mapPoint(fig, gs[1, 0], lat[indS], lon[indS], x, cb=True)
ax2.set_title(a)
fig.show()
