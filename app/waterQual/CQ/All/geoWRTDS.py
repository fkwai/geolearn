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
n = 40*5
code = '00915'
bMat = countMat[:, codeCount.index(code)] > n
indSel = np.where(countMat[:, codeCount.index(code)] > n)[0]
siteNoLst = [siteNoLstAll[ind] for ind in indSel]
dfG = dfGeo.loc[siteNoLst]

# pca by group
dictVar = gageII.getVariableDict()
# grpLst = list(dictVar.keys())
grpLst = ['LC06_Basin', 'LC06_Mains100', 'LC_Crops']
matPca = np.ndarray([len(siteNoLst), len(grpLst), 10])
for k, grp in enumerate(grpLst):
    varG = list(set(dfG.columns.tolist()).intersection(dictVar[grp]))
    npca = min(len(varG), 10)
    x = dfG[varG].values
    x[np.isnan(x)] = -1
    pca = decomposition.PCA(n_components=npca)
    pca.fit(x)
    xx = pca.transform(x)
    matPca[:, k, :npca] = xx

# plot
corrMat = dfCorr.loc[siteNoLst][code].values
for k, grp in enumerate(grpLst):
    fig, ax = plt.subplots(1, 1)
    ax.plot(matPca[:, k, 0], corrMat, '*')
    ax.set_title(grp)
    fig.show()
