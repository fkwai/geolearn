from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time

# pick out sites that are have relative large number of observations
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
df = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                 dtype={'siteNo': str}).set_index('siteNo')

# pick some sites
code = '00915'
siteNoLst = df[df[code] > 1000].index.tolist()
nSite = df.loc[siteNoLst][code].values
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(8, 4))
    axplot.mapPoint(axM, lat, lon, nSite, s=12)
    figP, axP = plt.subplots(1, 1, figsize=(12, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfC = waterQuality.readSiteY(siteNo, [code])
    t = dfC.index.values.astype(np.datetime64)
    axplot.plotTS(axP, t, dfC[code], styLst='*')
    axP.set_title('{} #samples = {}'.format(siteNo, dfC.count().values))


figplot.clickMap(funcMap, funcPoint)

siteNo = '401733105392404'
dfC = waterQuality.readSiteY(siteNo, usgs.codeLst)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(dfC)
fig.show()
