
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
codeLst = ['00605', '00915']
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

# plot ts
mapLst = [corrMat[:, k] for k in range(nc)]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codeNameLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]


def funcMap():
    figM, axM = plt.subplots(1, nc, figsize=(12, 4))
    for k in range(nc):
        axplot.mapPoint(axM[k], lat, lon, mapLst[k], s=16)
        axM[k].set_title('WRTDS corr {}'.format(codeNameLst[k]))
    figP, axP = plt.subplots(len(codeLst), 1, figsize=[16, 6])
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    print(iP, siteNo)
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq='D')
    t = dfO.index
    for k, code in enumerate(codeLst):
        ax = axP[k]
        axplot.plotTS(ax, t, dfO[code] * dfO['00060'], styLst='*', cLst='k')


importlib.reload(axplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)
