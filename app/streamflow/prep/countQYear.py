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
dfCount = pd.DataFrame(index=yrLst, columns=siteNoLstAll).fillna(0)
sd = pd.datetime(1980, 1, 1)
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfC = dfQ.groupby(dfQ.index.year).agg('count')
    dfCount.update(dfC.rename(columns={'00060_00003': siteNo}))
np.save(os.path.join(dirInv, 'matCountQ'), dfCount.values)
