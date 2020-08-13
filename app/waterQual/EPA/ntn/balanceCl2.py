import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

dfCount = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                      dtype={'siteNo': str}).set_index('siteNo')

# pick some sites
code = '00945'
varC = [code]
varNtn = ['SO4']
siteNoLst = dfCount[dfCount[code] > 100].index.tolist()
nSite = dfCount.loc[siteNoLst][code].values

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLst)
dfCrd = gageII.updateCode(dfCrd)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


# add a start/end date to improve efficiency.
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
sd = t[0]
ed = t[-1]
td = pd.date_range(sd, ed)
ntnFolder = os.path.join(kPath.dirData, 'EPA', 'NTN', 'usgs', 'weeklyRaw')
matRate = np.ndarray([len(siteNoLst), 2])

for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfF = gridMET.readBasin(siteNo)
    dfP = pd.read_csv(os.path.join(ntnFolder, siteNo), index_col='date')
    dfP = dfP[varNtn]
    # merge to one table
    df = pd.DataFrame({'date': td}).set_index('date')
    df = df.join(dfC)
    df = df.join(dfQ)
    df = df.join(dfF)
    df = df.rename(columns={'00060_00003': '00060'})
    # convert to weekly
    dfW = df.resample('W-TUE').mean()
    dfW = dfW.join(dfP)

    temp1 = dfW['00945']/dfW['SO4']
    matRate[k, 0] = temp1.mean()

figM, axM = plt.subplots(1, 1, figsize=(8, 4))
axplot.mapPoint(axM, lat, lon, matRate[:, 0], s=12)
figM.show()
