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
import scipy
import json

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
# countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))


# sum up
# for code in codeLst:
code = '00600'
ic = codeLst.index(code)
count = countMatW[:, :, ic]
nyLst = list(range(20))
nsLst = list(range(20))
mat = np.ndarray([len(nsLst), len(nyLst)])
for j, ns in enumerate(nsLst):
    for i, ny in enumerate(nyLst):
        mat[j, i] = len(np.where(np.sum(count > ns, axis=1) > ny)[0])
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# axplot.plotHeatMap(ax, mat, labLst=list(range(20)))
# fig.show()
grad = np.gradient(mat)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
axplot.plotHeatMap(ax, (grad[0]+grad[1])/2/mat*100, labLst=list(range(20)))
ax.set_title(code)
# axplot.plotHeatMap(ax, (grad[1])/mat*100, labLst=list(range(20)))
fig.show()

# select sites - 6 yrs > 10 samples or 2 yrs > 20 samples
indLst = list()
for code in codeLst:
    ic = codeLst.index(code)
    ny = 6
    ns = 10
    count = countMatW[:, :, ic]
    indS1 = np.where(np.sum(count > ns, axis=1) > ny)[0]
    ny = 2
    ns = 20
    indS2 = np.where(np.sum(count > ns, axis=1) > ny)[0]
    indS = np.unique(np.concatenate([indS1, indS2]))
    # indS = indS1
    a = len(np.setdiff1d(indS2, indS1))
    b = len(indS)
    print('{} {} {}'.format(code, a, b))
    indLst.append(indS)
indAll = np.unique(np.concatenate(indLst[2:]))
len(indAll)

# save to dict
dictSite = dict()
for k, code in enumerate(codeLst):
    siteNoLst = [siteNoLstAll[ind] for ind in indLst[k]]
    dictSite[code] = siteNoLst
dictSite['comb'] = [siteNoLstAll[ind] for ind in indAll]
saveName = os.path.join(dirInv, 'dictStableSites_0610_0220')
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)

# plot sites
code = '00600'
siteNoLst = dictSite[code]
indS = indLst[codeLst.index(code)]
lat = dfCrd.loc[siteNoLst]['LAT_GAGE'].values
lon = dfCrd.loc[siteNoLst]['LNG_GAGE'].values
nSite = np.sum(count[indS, :], axis=1)


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
