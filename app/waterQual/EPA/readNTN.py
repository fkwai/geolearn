from hydroDL.data import usgs, gageII
from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import datetime

dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')

tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)

# fix the tabel
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)

# siteNoLst
siteIdLst1 = tabData['siteID'].unique().tolist()
siteIdLst2 = tabSite['siteid'].tolist()
siteIdLst = list(set(siteIdLst1).intersection(siteIdLst2))

# start
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
varLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
flagLst = ['flagCa', 'flagMg', 'flagK', 'flagNa', 'flagNH4',
           'flagNO3', 'flagCl', 'flagSO4', 'valcode', 'invalcode']
# read site
siteId = 'NC03'
tab = tabData[tabData['siteID'] == 'NC03']
tab.index = pd.to_datetime(tab['dateon'])
weekday = tab.index.normalize().weekday
tab2 = pd.DataFrame(index=t)
tol = pd.Timedelta(3, 'D')
tab2 = pd.merge_asof(left=tab2, right=tab, right_index=True,
                     left_index=True, direction='nearest', tolerance=tol)
tab2.to_csv('temp')
weekday.value_counts(normalize=True)
weekday.value_counts()

tab['weekday'] = weekday
tab[weekday > 2]

tab = tabData[tabData['siteID'] == 'NY68']
tab = tab.replace(-9, np.nan)
fig, ax = plt.subplots(1, 1)
ax.plot(tab['SO4'],'*')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(tabData[tabData['siteID'] == 'KS31']['SO4'])
ax.plot(tabData[tabData['siteID'] == 'KS32']['SO4'])
ax.plot(tabData[tabData['siteID'] == 'KS97']['SO4'])
ax.plot(tabData[tabData['siteID'] == 'KS07']['SO4'])
fig.show()


varLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
flagLst = ['flagCa', 'flagMg', 'flagK', 'flagNa', 'flagNH4',
           'flagNO3', 'flagCl', 'flagSO4', 'valcode', 'invalcode']


tab = tabData[tabData['siteID'] == 'NY68']
