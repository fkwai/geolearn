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

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)


# extract a 4d matrix of data count
yrLst = list(range(1980, 2020))
countMatD = np.full([len(siteNoLstAll), len(yrLst), len(codeLst)], 0)
countMatW = np.full([len(siteNoLstAll), len(yrLst), len(codeLst)], 0)
sd = pd.datetime(1980, 1, 1)
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfCountD = pd.DataFrame(index=yrLst, columns=codeLst).fillna(0)
    dfCountW = pd.DataFrame(index=yrLst, columns=codeLst).fillna(0)
    dfC = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd)
    dfCW = dfC.resample('W-TUE').mean()
    dfCountD.update(dfC.groupby(dfC.index.year).agg('count'))
    dfCountW.update(dfCW.groupby(dfCW.index.year).agg('count'))
    countMatD[k, :, :] = dfCountD.values
    countMatW[k, :, :] = dfCountW.values

np.save(os.path.join(dirInv, 'matCountDaily'), countMatD)
np.save(os.path.join(dirInv, 'matCountWeekly'), countMatW)

# sum up
nyLst = list(range(5, 21))  # samples in year
dfCountD = pd.DataFrame(index=nyLst, columns=codeLst).fillna(0)

for code in codeLst:
    ic = codeLst.index(code)
    countD = countMatD[:, :, ic]
    for ny in nyLst:
        # dfCountD.loc[ny][code] = len(np.where(countD > ny)[0])
        dfCountD.loc[ny][code] = len(np.where(countD > ny)[0])
