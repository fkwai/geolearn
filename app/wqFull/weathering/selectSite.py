import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# count - only for C now
saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
npz = np.load(saveFile)
matC = npz['matC']
matCF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])

# devide train / test
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
t1 = np.datetime64('1982-01-01', 'D')
t2 = np.datetime64('2010-01-01', 'D')
t3 = np.datetime64('2019-01-01', 'D')
indT1 = np.where((tR >= t1) & (tR < t2))[0]
indT2 = np.where((tR >= t2) & (tR < t3))[0]
indC = [codeLst.index(code) for code in codeSel]

matB = matC & (~matCF)
mat = matB[:, :, indC]
mat1 = matB[:, indT1, :][:, :, indC]
mat2 = matB[:, indT2, :][:, :, indC]
count = np.sum(np.all(mat, axis=2), axis=1)
count1 = np.sum(np.all(mat1, axis=2), axis=1)
count2 = np.sum(np.all(mat2, axis=2), axis=1)

# pick
pickMat = (count >= 400)
len(np.where(pickMat)[0])
indS = np.where(pickMat)[0]
dictSite = dict()
siteNoSel = [siteNoLst[ind] for ind in indS]

siteNoSel = ['01184000', '01434025', '01435000', '01466500', '04063700', '06313500',
             '06317000', '06324500', '09163500', '09352900', '11264500', '401733105392404']
indS = [siteNoLst.index(siteNo) for siteNo in siteNoSel]
dictSite['k12'] = siteNoSel


dfCrd = gageII.readData(siteNoLst=siteNoSel, varLst=[
                        'DRAIN_SQKM', 'LNG_GAGE', 'LAT_GAGE'])
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
area = dfCrd['DRAIN_SQKM'].values
nc = len(codeSel)


def funcM():
    figM, axM = plt.subplots(2, 1, figsize=(6, 4))
    axplot.mapPoint(axM[0], lat, lon, area, s=16, cb=True)
    axplot.mapPoint(axM[1], lat, lon, count[indS], s=16, cb=True)
    figP, axP = plt.subplots(nc, 1, figsize=(12, 8))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoSel[iP]
    df = dbBasin.readSiteTS(siteNo, varLst=codeSel, freq='D')
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    axplot.multiTS(axP, df.index.values, df[codeSel].values, labelLst=codeSel)
    figP.subplots_adjust(hspace=0)
    figP.suptitle('{} {}'.format(siteNo, area))


figM, figP = figplot.clickMap(funcM, funcP)

# save
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
saveName = os.path.join(dirInv, 'siteSel', 'dictWeathering')
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)
