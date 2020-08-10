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

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)

dfCountYr = pd.read_csv(os.path.join(
    dirInv, 'siteCountWeekly-Y10.csv'), dtype={'siteNo': str})
dfCountYr = dfCountYr.set_index('siteNo')
temp = dfCountYr[codeLst[2:]] >= 6
temp.sum(axis=1).value_counts().sort_index(ascending=False).cumsum()
tempSum = temp.sum(axis=1)
siteSel = tempSum.index[tempSum >= 16]
# dfCountYr.loc[siteSel].to_csv('temp.csv')

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
lat = dfCountYr.loc[siteSel]['lat'].values
lon = dfCountYr.loc[siteSel]['lon'].values
data = tempSum[siteSel].values
axplot.mapPoint(ax, lat, lon, data, s=12)
fig.show()
