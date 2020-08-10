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

columns = ['lat', 'lon', 'ref']+codeLst
dfCount = pd.DataFrame(index=siteNoLstAll, columns=columns)
dfCount['lat'] = dfCrd['LAT_GAGE']
dfCount['lon'] = dfCrd['LNG_GAGE']
dfCount['ref'] = dfCrd['CLASS']
dfCount.index.name = 'siteNo'
dfCountY10 = dfCount.copy()
dfCountY15 = dfCount.copy()
dfCountY20 = dfCount.copy()
sd = pd.datetime(1980, 1, 1)
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfC = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd)
    dfCW = dfC.resample('W-TUE').mean()
    dfCount = dfCount.set_value(siteNo, codeLst, dfC.count())
    dfYr = dfCW.groupby(dfCW.index.year).agg('count')
    dfCountY10 = dfCountY10.set_value(siteNo, codeLst, (dfYr >= 10).sum())
    dfCountY15 = dfCountY15.set_value(siteNo, codeLst, (dfYr >= 15).sum())
    dfCountY20 = dfCountY20.set_value(siteNo, codeLst, (dfYr >= 20).sum())
dfCount.to_csv(os.path.join(dirInv, 'siteCountWeekly.csv'))
dfCountY10.to_csv(os.path.join(dirInv, 'siteCountWeekly-Y10.csv'))
dfCountY15.to_csv(os.path.join(dirInv, 'siteCountWeekly-Y15.csv'))
dfCountY20.to_csv(os.path.join(dirInv, 'siteCountWeekly-Y20.csv'))

# app\waterQual\EPA\countSiteWeek.py
# dfCountYr = pd.read_csv(os.path.join(
#     dirInv, 'siteCountWeekly-Y15.csv'), dtype={'siteNo': str})
# dfCountYr = dfCountYr.set_index('siteNo')


# temp = dfCountYr[codeLst[2:]] >= 6
# temp.sum(axis=1).value_counts().sort_index(ascending=False).cumsum()
# siteSel = temp.index[temp.sum(axis=1) >= 16]
# dfCountYr.loc[siteSel].to_csv('temp.csv')

# figM, axM = plt.subplots(1, 1, figsize=(8, 4))
# axplot.mapPoint(axM, lat, lon, nSite, s=12)
# figP, axP = plt.subplots(1, 1, figsize=(12, 6))
