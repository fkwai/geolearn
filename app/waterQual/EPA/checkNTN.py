from hydroDL.data import usgs, gageII, ntn
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

# check if all fields are available
varPLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
tab = tabData[varPLst]
tab = tab.dropna(how='all')

# recheck weekday - same conclusion
ind = tab.index.values
t1 = pd.to_datetime(tabData['dateon'],
                    infer_datetime_format=True).dt.normalize()
t2 = pd.to_datetime(tabData['dateoff'],
                    infer_datetime_format=True).dt.normalize()
wd1 = t1[ind].dt.weekday
wd2 = t2[ind].dt.weekday
wd1.value_counts(normalize=True)
wd2.value_counts(normalize=True)

# check for distributions
dataName = 'refWeek'
wqData = waterQuality.DataModelWQ(dataName)
for var in ntn.varLst+['distNTN']:
    v = wqData.f[:, :, wqData.varF.index(var)]
    fig, axes = plt.subplots(2, 2)
    temp = v.flatten()
    temp90 = temp[np.where((temp > np.nanpercentile(temp, 5)) &
                           (temp < np.nanpercentile(temp, 95)))]
    _ = axes[0, 0].hist(temp, bins=100)
    _ = axes[0, 1].hist(temp90, bins=100)
    try:
        _ = axes[1, 0].hist(np.log(temp+1), bins=100)
        _ = axes[1, 1].hist(np.log(temp90+1), bins=100)
    except(ValueError):
        print(var+' can not log')
    fig.suptitle(var)
    fig.show()
