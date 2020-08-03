from hydroDL import kPath, utils
from hydroDL.data import gageII, ntn
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
siteIdLst = tabData['siteID'].unique().tolist()
varLst = ntn.varLst

# time index
t = pd.date_range(start='1979-01-01', end='2019-12-31', freq='W-TUE')
t1 = t[:-1]
t2 = t[1:]  # use as index

# transfer to roughly weekly
saveDir = os.path.join(dirNTN, 'weeklyRaw')
for k, siteId in enumerate(siteIdLst):
    print(k,siteId)
    tab = tabData[tabData['siteID'] == siteId]
    tab.index = pd.to_datetime(tab['dateoff'])
    out = pd.DataFrame(index=t2)
    tol = pd.Timedelta(3, 'D')
    out = pd.merge_asof(left=out, right=tab, right_index=True,
                        left_index=True, direction='nearest', tolerance=tol)
    dfOut = out[varLst]
    dfOut.index.name = 'date'
    dfOut.to_csv(os.path.join(saveDir, siteId))
