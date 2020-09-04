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
df0 = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df1 = pd.read_csv(os.path.join(dirInv, 'codeCount_B2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df2 = pd.read_csv(os.path.join(dirInv, 'codeCount_A2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')

# pick some sites
code = '00955'
siteNoLst = df0[(df1[code] > 100) & (df2[code] > 100)].index.tolist()
nB = df1.loc[siteNoLst][code].values
nA = df2.loc[siteNoLst][code].values
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(8, 4))
    axplot.mapPoint(axM[0], lat, lon, nB, s=12)
    axplot.mapPoint(axM[1], lat, lon, nA, s=12)
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfC = waterQuality.readSiteY(siteNo, [code])
    t = dfC.index.values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    axplot.plotTS(axP, t, dfC[code], styLst='*', tBar=tBar)
    n1 = dfC[dfC[code].index < tBar].count().values
    n2 = dfC[dfC[code].index >= tBar].count().values
    axP.set_title('{} #samples = {} {}'.format(siteNo, n1, n2))


figplot.clickMap(funcMap, funcPoint)
