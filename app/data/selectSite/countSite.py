from hydroDL import kPath
from hydroDL.data import usgs
import pandas as pd
import numpy as np
import os
import datetime as dt
import time

# all gages
t0 = time.time()
siteNoFile = os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite = pd.read_csv(siteNoFile, dtype={'siteNo': str})
siteNoLst = dfSite['siteNo'].tolist()
dirInv = os.path.join(kPath.dirUsgs, 'siteSel')

# extract bool matrix of availble measurements
sd = dt.datetime(1979, 1, 1)
ed = dt.datetime(2022, 12, 31)
t = pd.date_range(sd, ed)
codeLst = sorted(usgs.varC)
flagLst = [code + '_cd' for code in codeLst]
# matC [#t, #site, #varC]
matC = np.ndarray([len(t), len(siteNoLst), len(codeLst)], dtype=bool)
matF = np.ndarray([len(t), len(siteNoLst), len(codeLst)], dtype=bool)
for iS, siteNo in enumerate(siteNoLst):
    print('C', iS, siteNo, time.time() - t0)
    codeLst = usgs.varC
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd, flag=2)
    if dfC is not None:
        dfCount = pd.DataFrame(index=t, columns=codeLst).fillna(False)
        dfCount.update(~dfC.isna())
        dfCountFlag = pd.DataFrame(index=t, columns=flagLst).fillna(False)
        dfCF = dfCF.fillna(False)
        dfCF = dfCF.astype(bool)
        dfCountFlag.update(dfCF)
        matC[:, iS, :] = dfCount.values.astype(bool)
        matF[:, iS, :] = dfCountFlag.values.astype(bool)

# matQ [#t, #site]
matQ = np.ndarray([len(t), len(siteNoLst)], dtype=bool)
for iS, siteNo in enumerate(siteNoLst):
    print('Q', iS, siteNo, time.time() - t0)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    if dfQ is not None:
        dfCount = pd.DataFrame(index=t, columns=['00060_00003']).fillna(False)
        dfCount.update(~dfQ.isna())
        mat = dfCount.values.astype(bool)
        matQ[:, iS] = mat[:, 0]

outFile = os.path.join(dirInv, 'matBool')
np.savez_compressed(
    outFile, matC=matC, matQ=matQ, matF=matF, siteNoLst=siteNoLst, t=t, codeLst=codeLst
)
