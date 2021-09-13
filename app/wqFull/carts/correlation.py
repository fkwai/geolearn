
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

matCorr = np.corrcoef(dfG.values, rowvar=False)
ns, ng = dfG.shape

fig, ax = plt.subplots(1, 1)
ax.imshow(matCorr, vmin=0.5, vmax=1)
fig.show()

# group those varibles
th = 0.9
grpLst = list()
for k in range(ng):
    ind = np.where(matCorr[k, :] > th)[0]
    if len(ind) > 1:
        grpLst.append(tuple(ind))
grpSet = set(grpLst)


# nan values
colNan = dfG.columns[dfG.isna().any()].tolist()
dfG[colNan]
