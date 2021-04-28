from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# load all site counts
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
countD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'], siteNoLst=siteNoLstAll)

# find sites for code
codeP = ['00915', '00925']
nc = len(codeP)
ic = [codeLst.index(code) for code in codeP]
countP = np.sum(countD[:, :, ic], axis=1)
bMat = np.all(countP > 200, axis=1)
ind = np.where(bMat)[0]
siteNoLst = [siteNoLstAll[x] for x in ind]
count = countP[ind, :]
lat = dfCrd.loc[siteNoLst]['LAT_GAGE']
lon = dfCrd.loc[siteNoLst]['LNG_GAGE']


def funcM():
    figM, axM = plt.subplots(nc, 1, figsize=(6, 4))
    for k, code in enumerate(codeP):
        axplot.mapPoint(axM[k], lat, lon, count[:, k], s=16, cb=True)
    figP, axP = plt.subplots(nc, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoLst[iP]
    df = dbBasin.readSiteTS(siteNo, varLst=codeP, freq='D')
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    for k, code in enumerate(codeP):
        axplot.plotTS(axP[k], df.index, df[code].values, cLst='k')
        axP[k].set_title('{} of {} {}'.format(code, siteNo, area))


figM, figP = figplot.clickMap(funcM, funcP)
