
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

# select site
n = 40*2
codeLst = ['00600', '00915']
nameLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
nc = len(codeLst)
icLst = [codeCount.index(code) for code in codeLst]
bMat = countMat[:, icLst] > n
# indSel = np.where(bMat.any(axis=1))
indSel = np.where(bMat.all(axis=1))[0]
siteNoLst = [siteNoLstAll[ind] for ind in indSel]

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-D', 'All')
fileCorr = os.path.join(dirWrtds, 'corr')
dfCorr = pd.read_csv(fileCorr, dtype={'siteNo': str}).set_index('siteNo')
corrMat = dfCorr.loc[siteNoLst][codeLst].values

# eco region
dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinEco')
dfEcoAll = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')
dfEco = dfEcoAll.loc[siteNoLst]

# plot box
for field in ['code'+str(k) for k in range(3)]:
    dfEco[field] = dfEco[field].astype(int).astype(str).str.zfill(2)
dfEco['comb'] = dfEco[['code0', 'code1']].agg('-'.join, axis=1)
ecoLst = sorted(dfEco['comb'].unique().tolist())
dataBox = list()
for eco in ecoLst:
    temp = list()
    for k in range(nc):
        ind = np.where((dfEco['comb'] == eco).values)[0]
        temp.append(corrMat[ind, k])
    dataBox.append(temp)
label1 = ecoLst
label2 = nameLst
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(12, 4), yRange=[0, 1])
fig.show()
