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
siteNoFile = os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite = pd.read_csv(siteNoFile, dtype={'siteNo': str})
siteNoLst = dfSite['siteNo'].tolist()
codeLst = sorted(usgs.varC)
dirInv = os.path.join(kPath.dirUsgs, 'siteSel')

# extract a 3d matrix [#t, #site, #code] of data count
sd = pd.datetime(1979, 1, 1)
ed= pd.datetime(2023, 1, 1)
dfCount=pd.DataFrame(index=siteNoLst, columns=codeLst).fillna(0)
for siteNo in siteNoLst:    
    dfC = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd)    
    if dfC is not None:
        dfC = dfC[dfC.index < ed]
        dfCount.loc[siteNo].update(dfC.count())

# extract a 4d matrix of data count
yrLst = list(range(1979, 2023))
countMat = np.full([len(siteNoLst), len(yrLst), len(codeLst)], 0)
sd = pd.datetime(1979, 1, 1)
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfCountD = pd.DataFrame(index=yrLst, columns=codeLst).fillna(0)
    dfCountW = pd.DataFrame(index=yrLst, columns=codeLst).fillna(0)
    dfC = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd)    
    dfCountD.update(dfC.groupby(dfC.index.year).agg('count'))    
    countMat[k, :, :] = dfCountD.values    

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
