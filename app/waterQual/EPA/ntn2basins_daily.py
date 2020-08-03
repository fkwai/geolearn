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

# read NTN
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)

# transfer to weekly
t1 = pd.to_datetime(tabData['dateon'],
                    infer_datetime_format=True).dt.normalize()
t2 = pd.to_datetime(tabData['dateoff'],
                    infer_datetime_format=True).dt.normalize()
wd1 = t1.dt.weekday
wd2 = t2.dt.weekday

ind = np.where((wd1 == 1) & (wd2 == 1))[0]

# pick out sites that are have relative large number of observations
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# gageII
varG = ['LAT_GAGE', 'LNG_GAGE', 'CLASS']
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLstAll)
tabG = gageII.updateCode(tabG)

siteNo = '01013500'
lat = tabG.loc[siteNo]['LAT_GAGE']
lon = tabG.loc[siteNo]['LNG_GAGE']

# NTN
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)

latNtn = tabSite['latitude'].values
lonNtn = tabSite['longitude'].values

ind = np.argmin(np.sqrt((latNtn - lat)**2 + (lonNtn - lon)**2))
idNtn = tabSite.iloc[ind]['siteid']

t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')

tt = t[0]
t1 = pd.to_datetime(tabData['dateon'],
                    infer_datetime_format=True).dt.normalize()
t2 = pd.to_datetime(tabData['dateoff'],
                    infer_datetime_format=True).dt.normalize()
tabData[(t1 <= tt) & (t2 >= tt)]
