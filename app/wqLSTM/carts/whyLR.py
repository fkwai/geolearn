
import scipy
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sklearn.tree
import matplotlib.gridspec as gridspec
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC
siteNoLst = DF.siteNoLst

# gageII
dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)
dfG = gageII.removeField(dfG)
colLst = dfG.columns

# load linear
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param'
matQ = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matQ[:, k] = dfCorr['rsq'].values

# load seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param'
matS = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matS[:, k] = dfCorr['rsq'].values

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matRm = count < 100
matQ[matRm] = np.nan
matS[matRm] = np.nan


# calculate correlation
dfRQ = pd.DataFrame(index=colLst, columns=codeLst)
dfRS = pd.DataFrame(index=colLst, columns=codeLst)
for ic, code in enumerate(codeLst):
    [rq], indq = utils.rmNan([matQ[:, ic]])
    [rs], inds = utils.rmNan([matS[:, ic]])
    for j, col in enumerate(colLst):
        r1 = scipy.stats.spearmanr(rq, dfG.values[indq, j])[0]
        r2 = scipy.stats.spearmanr(rs, dfG.values[inds, j])[0]
        dfRQ.at[col, code] = r1
        dfRS.at[col, code] = r2

# find correlated vars
outDir = r'C:\Users\geofk\work\waterQuality\statAttr'
dfRQ.to_csv(os.path.join(outDir, 'attrLinear.csv'))
dfRS.to_csv(os.path.join(outDir, 'attrSeason.csv'))

dfRQ2 = dfRQ**2
dfRS2 = dfRS**2
dfRQ2.to_csv(os.path.join(outDir, 'attrLinear2.csv'))
dfRS2.to_csv(os.path.join(outDir, 'attrSeason2.csv'))

np.max(dfRQ.values.flatten())
np.min(dfRQ.values.flatten())

np.max(dfRS.values.flatten())
np.min(dfRS.values.flatten())

ind1, ind2 = np.where(abs(dfRQ.values) > 0.4)
col = colLst[ind1]
code = set([codeLst[x] for x in ind2])

# for single code
code = '00935'
indC = codeLst.index(code)
ind1 = np.where(abs(dfRQ[code].values) > 0.4)[0]
dfRQ.iloc[ind1][code]

code = '00915'
indC = codeLst.index(code)
var = 'CONTACT'
fig, ax = plt.subplots(1, 1)
ax.plot(dfG[var].values, matS[:, indC], '*')
fig.show()
