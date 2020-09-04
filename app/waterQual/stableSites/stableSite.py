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
countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

ns = 12
ny = 4
indLst = list()
for code in codeLst:
    ic = codeLst.index(code)
    count = countMatW[:, :, ic]
    indS = np.where(np.sum(count > ns, axis=1) > ny)[0]
    indLst.append(indS)
    print('{} {}'.format(code, len(indS)))
# get rid of 00010 and 00095
indAll = np.unique(np.concatenate(indLst[2:]))
print(len(indAll))
indAll = np.unique(np.concatenate(indLst[1:]))
print(len(indAll))
indAll = np.unique(np.concatenate(indLst))
print(len(indAll))

# save to dict
dictSite = dict()
for k, code in enumerate(codeLst):
    siteNoLst = [siteNoLstAll[ind] for ind in indLst[k]]
    dictSite[code] = siteNoLst
for k in range(3):
    indComb = np.unique(np.concatenate(indLst[k:]))
    # comb0 - all; comb1 - rm 00010; comb2 - rm 00095
    dictSite['comb'+str(k)] = [siteNoLstAll[ind] for ind in indAll]
saveName = os.path.join(dirInv, 'siteSel', 'dictSB_0412')
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)

# plot sites
# code = '00600'
# siteNoLst = dictSite[code]
# dfCrd = gageII.readData(
#     varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
# dfCrd = gageII.updateCode(dfCrd)
# indS = indLst[codeLst.index(code)]
# lat = dfCrd.loc[siteNoLst]['LAT_GAGE'].values
# lon = dfCrd.loc[siteNoLst]['LNG_GAGE'].values
# nSite = np.sum(count[indS, :], axis=1)


# def funcMap():
#     figM, axM = plt.subplots(1, 1, figsize=(8, 4))
#     axplot.mapPoint(axM, lat, lon, nSite, s=12)
#     figP, axP = plt.subplots(1, 1, figsize=(12, 6))
#     return figM, axM, figP, axP, lon, lat


# def funcPoint(iP, axP):
#     siteNo = siteNoLst[iP]
#     dfC = waterQuality.readSiteY(siteNo, [code])
#     t = dfC.index.values.astype(np.datetime64)
#     axplot.plotTS(axP, t, dfC[code], styLst='*')
#     axP.set_title('{} #samples = {}'.format(siteNo, dfC.count().values))


# figplot.clickMap(funcMap, funcPoint)

# two crit - 6 yrs > 10 samples or 2 yrs > 20 samples
# indLst = list()
# for code in codeLst:
#     ic = codeLst.index(code)
#     ny = 6
#     ns = 20
#     count = countMatW[:, :, ic]
#     # count = countMatD[:, :, ic]
#     indS1 = np.where(np.sum(count > ns, axis=1) > ny)[0]
#     ny = 2
#     ns = 20
#     indS2 = np.where(np.sum(count > ns, axis=1) > ny)[0]
#     indS = np.unique(np.concatenate([indS1, indS2]))
#     # indS = indS1
#     a = len(np.setdiff1d(indS2, indS1))
#     b = len(indS)
#     # print('{} {} {}'.format(code, a, b))
#     print('{} {}'.format(code, b))
#     indLst.append(indS)
# indAll = np.unique(np.concatenate(indLst[2:])) # get rid of 00010 and 00095
# len(indAll)
