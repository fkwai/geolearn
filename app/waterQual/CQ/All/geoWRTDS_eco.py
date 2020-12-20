import matplotlib.cm as cm
from sklearn import decomposition
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy

# count
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeCount = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
countMatAll = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))
countMat = np.ndarray([len(siteNoLstAll), len(codeCount)])
for ic, code in enumerate(codeCount):
    countMat[:, ic] = np.sum(countMatAll[:, :, ic], axis=1)


# load WRTDS performance
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-D', 'All')
fileC = os.path.join(dirWrtds, 'corr')
dfCorr = pd.read_csv(fileC, dtype={'siteNo': str}).set_index('siteNo')
fileR = os.path.join(dirWrtds, 'rmse')
dfRmse = pd.read_csv(fileR, dtype={'siteNo': str}).set_index('siteNo')

# load geo attr
dfGeo = gageII.readData(siteNoLst=siteNoLstAll)
dfGeo = gageII.updateCode(dfGeo)

# select site
n = 40*2
code = '00915'
bMat = countMat[:, codeCount.index(code)] > n
indSel = np.where(countMat[:, codeCount.index(code)] > n)[0]
siteNoLst = [siteNoLstAll[ind] for ind in indSel]
dfG = dfGeo.loc[siteNoLst]

# eco region
dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinEco')
dfEcoAll = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')
dfEco = dfEcoAll.loc[siteNoLst]
for field in ['code'+str(k) for k in range(3)]:
    dfEco[field] = dfEco[field].astype(int).astype(str).str.zfill(2)
dfEco['comb'] = dfEco[['code0', 'code1']].agg('-'.join, axis=1)
ecoLst = sorted(dfEco['comb'].unique().tolist())

# plot
geoField = 'FORESTNLCD06'
# geoField = 'PLANTNLCD06'
corrMat = dfCorr.loc[siteNoLst][code].values
geoMat = dfG[geoField].values
cLst = cm.rainbow(np.linspace(0, 1, len(ecoLst)))
fig, ax = plt.subplots(1, 1)
for k, eco in enumerate(ecoLst):
    ind = np.where((dfEco['comb'] == eco).values)[0]
    ax.plot(geoMat[ind], corrMat[ind], c=cLst[k],
            label=eco, marker='*', ls='')
ax.legend()
fig.show()

# plot
geoField = 'FORESTNLCD06'
# geoField = 'PLANTNLCD06'
corrMat = dfCorr.loc[siteNoLst][code].values
geoMat = dfG[geoField].values
eco = '10-01'
fig, ax = plt.subplots(1, 1)
ind = np.where((dfEco['comb'] == eco).values)[0]
ax.plot(geoMat[ind], corrMat[ind],'*')
fig.show()
