from hydroDL.data import usgs, gageII
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')

tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)

# week of day - 95% tuesday, 99% mon - wed
dateOn = tabData['dateon']
# dateOn = pd.to_datetime(tabData['dateon']).values.astype('datetime64[D]')
# weekday = pd.to_datetime(tabData['dateon']).dt.weekday
# weekday.value_counts(normalize=True)
# weekday.value_counts()

# siteID - site AB36 is missing but only has 4 samples; NC30 CO83 WI19 do not have data
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
siteIdLst = tabData['siteID'].unique().tolist()
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
set(lst2)-set(siteIdLst)
set(siteIdLst)-set(lst2)

# flags
dirCsv = os.path.join(dirNTN, 'csv')
tabData.columns
tabData['valcode'].unique()
set(''.join(tabData['invalcode'].unique().tolist()))

# startdate
pd.DatetimeIndex(tabSite['startdate']).year.value_counts()

# maps
dateOn = pd.to_datetime(tabData['dateon']).values.astype('datetime64[D]')
tabData = tabData.replace(-9, np.nan)

var = 'K'
tStrLst = ['1980-01-01', '1990-01-02', '2000-01-04', '2010-01-05']
fig, axes = plt.subplots(2, 2)
for k, tStr in enumerate(tStrLst):
    t = np.datetime64(tStr)
    ind = np.where(dateOn == t)
    data = tabData.iloc[ind][var].values
    tempSite = tabSite[tabSite['siteid'].isin(tabData.iloc[ind]['siteID'])]
    lat = tempSite['latitude'].values
    lon = tempSite['longitude'].values
    j, i = utils.index2d(k, 2, 2)
    # axplot.mapPoint(axes[j,i], lat, lon, data,vRange=[0,2.5])
    axplot.mapPoint(axes[j, i], lat, lon, data)
    axes[j, i].set_title('{} {}'.format(var, tStr))
fig.show()
