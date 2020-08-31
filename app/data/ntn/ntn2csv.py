from hydroDL import kPath, utils
from hydroDL.data import ntn
import pandas as pd
import numpy as np
import os
import time
import importlib

# save ntn data to csv files
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
tabData = ntn.readDataRaw()
ntnIdLst = tabData['siteID'].unique().tolist()
varLst = ntn.varLst

td = pd.date_range(start='1979-01-01', end='2019-12-31')
tw = pd.date_range(start='1979-01-01', end='2019-12-31', freq='W-TUE')
ntnFolderD = os.path.join(dirNTN, 'csv', 'daily')
ntnFolderW = os.path.join(dirNTN, 'csv', 'weekly')
tt0 = time.time()
for kk, ntnId in enumerate(ntnIdLst):
    tt1 = time.time()
    tab = tabData[tabData['siteID'] == ntnId]
    dfD = pd.DataFrame(index=td, columns=varLst, dtype=np.float32)
    dfW = pd.DataFrame(index=tw, columns=varLst, dtype=np.float32)
    for k in range(len(tab)):
        t1 = pd.to_datetime(tab.iloc[k]['dateon']).date()
        t2 = pd.to_datetime(tab.iloc[k]['dateoff']).date()
        tt = pd.date_range(t1, t2)[1:]
        data = np.tile(tab.iloc[k][varLst].values.astype(
            np.float32), [len(tt), 1])
        tabTemp = pd.DataFrame(index=tt, columns=varLst, data=data)
        dfD.update(tabTemp)
    dfDW = dfD.resample('W-TUE').mean()
    dfW.update(dfDW)
    dfD.index.name = 'date'
    dfW.index.name = 'date'
    dfD.to_csv(os.path.join(ntnFolderD, ntnId))
    dfW.to_csv(os.path.join(ntnFolderW, ntnId))
    tt2 = time.time()
    print('{} {} {:.3f} {:.3f}'.format(kk, ntnId, tt2-tt1, tt2-tt0))
print('Done')


dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
tabData = pd.read_csv(fileData)
tab = tabData[tabData['siteID'] == 'WV18']
tab[varLst].astype('float32').to_csv('temp')
