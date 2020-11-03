from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy
import json

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
freq = 'W'
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

# select sites
count = np.sum(countMatW, axis=1)
code = '00955'
nS = 200
ic = codeLst.index(code)
ind = np.where(count[:, ic] > nS)[0]
siteNoLst = [siteNoLstAll[x] for x in ind]

# plot data
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
matMap = count[ind, ic]


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, matMap, s=16)
    figP, axP = plt.subplots(3, 1, figsize=(16, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfO = waterQuality.readSiteTS(siteNo, ['runoff', 'pr', code], freq=freq)
    t = dfO.index.values
    axplot.plotTS(axP[0], t, dfO['runoff'].values, styLst='-*', cLst='bgr')
    axplot.plotTS(axP[1], t, dfO['pr'].values, styLst='-*', cLst='bgr')
    axplot.plotTS(axP[2], t, dfO[code].values, styLst='*', cLst='bgr')
    r = np.nanmean(dfO['runoff'].values)/np.nanmean(dfO['pr'].values)*365/100    
    axP[0].set_title('{} {:.3f}'.format(siteNo, r))


figM, figP = figplot.clickMap(funcMap, funcPoint)
