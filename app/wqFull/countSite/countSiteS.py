import scipy
import time
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.master import basins
from hydroDL.data import gageII, usgs, gridMET
from hydroDL import kPath, utils
import os
import pandas as pd
import numpy as np
from hydroDL import kPath

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.newC)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
sd = np.datetime64('1979-01-01')

# load all data
dictC = dict()
dictCF = dict()
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd, flag=2)
    dictC[siteNo] = dfC
    dictCF[siteNo] = dfCF
dictQ = dict()
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    dictQ[siteNo] = dfQ

# app\waterQual\stableSites\countSiteYear.py

# create huge bool matrices
sdStr = '1979-01-01'
edStr = '2019-12-31'
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
matC = np.ndarray([len(siteNoLstAll), len(tR), len(codeLst)], dtype=bool)
matQ = np.ndarray([len(siteNoLstAll), len(tR)], dtype=bool)
matCF = np.ndarray([len(siteNoLstAll), len(tR), len(codeLst)], dtype=bool)

t0 = time.time()
for k, siteNo in enumerate(siteNoLstAll):
    print('\t site {}/{} {}'.format(k, len(siteNoLstAll), time.time()-t0))
    tempC = pd.DataFrame({'date': tR}).set_index('date').join(dictC[siteNo])
    tempCF = pd.DataFrame({'date': tR}).set_index('date').join(dictCF[siteNo])
    tempQ = pd.DataFrame({'date': tR}).set_index('date').join(dictQ[siteNo])
    matCF[k, :, :] = tempCF.values == 1
    matC[k, :, :] = ~np.isnan(tempC.values)
    matQ[k, :] = ~np.isnan(tempQ.values[:, 0])

# save
siteNoLst = siteNoLstAll
saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat')
np.savez_compressed(saveFile, matCF=matCF, matC=matC, matQ=matQ,
                    tR=tR, codeLst=codeLst, siteNoLst=siteNoLst)

# count - only for C now
saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
npz = np.load(saveFile)
matC = npz['matC']
matCF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = npz['codeLst']
siteNoLst = npz['siteNoLst']

rho = 365
matCount = np.ndarray([len(siteNoLst), len(codeLst), 365])
for k in range(len(siteNoLst)):

    pass
k = 0
ic = 3
tempC = matC[k, :, ic]
temp = np.convolve(tempC, np.ones(rho), mode='valid')
cc = temp[tempC[rho-1:]]
unique, count = np.unique(cc, return_counts=True)
countSum = count.cumsum()
matCount[k, ic, unique.astype(int)] = countSum


matCount[k, ic, :]

t0 = time.time()
bb = np.apply_along_axis(lambda m: np.convolve(
    m, np.ones(365), mode='valid'), axis=1, arr=matC)
print(time.time()-t0)
