import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


# varC = usgs.varC
varC = ['00940']
siteNoLst = ['0422026250', '04232050', '0423205010']

t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
t = t[:-1]

# read NTN
ntnId = 'NY43'
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabData = pd.read_csv(fileData)
tabSite = pd.read_csv(fileSite)
tabData['siteID'] = tabData['siteID'].apply(lambda x: x.upper())
tabData = tabData.replace(-9, np.nan)
tab = tabData[tabData['siteID'] == ntnId]
tab.index = pd.to_datetime(tab['dateon'])
weekday = tab.index.normalize().weekday
tab2 = pd.DataFrame(index=t)
tol = pd.Timedelta(3, 'D')
tab2 = pd.merge_asof(left=tab2, right=tab, right_index=True,
                     left_index=True, direction='nearest', tolerance=tol)
varPLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
dfP = tab2[varPLst]


siteNo = siteNoLst[2]
dfC = usgs.readSample(siteNo, codeLst=varC)
dfQ = usgs.readStreamflow(siteNo)
dfF = gridMET.readBasin(siteNo)

ntnFolder = os.path.join(kPath.dirData, 'EPA', 'NTN', 'usgs', 'weeklyRaw')
dfP = pd.read_csv(os.path.join(ntnFolder, siteNo), index_col='date')


# plot
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
ax.plot(dfP['Cl'], 'b--*')
ax.plot(dfP['cl'], 'g--*')
ax2.plot(dfC[varC], 'r*')


fig.show()

