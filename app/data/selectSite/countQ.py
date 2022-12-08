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
dirCsv = os.path.join(kPath.dirUSGS, 'streamflow', 'csv')
siteNoLst = [f for f in sorted(os.listdir(dirCsv))]

# extract a 4d matrix of data count
yrLst = list(range(1979, 2020))
dfCount = pd.DataFrame(index=yrLst, columns=siteNoLst).fillna(0)
sd = pd.datetime(1979, 1, 1)

for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfC = dfQ.groupby(dfQ.index.year).agg('count')
    dfCount.update(dfC.rename(columns={'00060_00003': siteNo}))
# np.save(os.path.join(dirInv, 'matCountQ'), dfCount.values)
countFile = os.path.join(kPath.dirUSGS, 'streamflow', 'countYr.csv')
dfCount.to_csv(countFile)
