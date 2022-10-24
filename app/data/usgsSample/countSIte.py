from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os

# site inventory
usgsDir = os.path.join(kPath.dirData, 'USGS')
invDir = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileInvC = os.path.join(invDir, 'inventory_NWIS_sample')
fileInvQ = os.path.join(invDir, 'inventory_NWIS_streamflow')
siteC = usgs.readUsgsText(fileInvC)
siteQ = usgs.readUsgsText(fileInvQ)
tabGageII = gageII.readTab('bas_classif')

siteNoC = siteC['site_no'].unique().tolist()
siteNoLenC = np.array([len(x) for x in siteNoC])

len(np.where(siteNoLenC == 8)[0])

temp = siteC[~(siteC['end_dt'] < pd.to_datetime('1980-01-01'))]
temp = temp[siteC['parm_cd'].isin(usgs.varC)]
temp = temp[(temp['count_nu'] > 1)]
len(temp['site_no'].unique())

tempC = temp


len(siteQ['site_no'].unique())
temp = siteQ[~(siteQ['qw_end_date'] < pd.to_datetime('1980-01-01'))]
len(temp['site_no'].unique())
tempQ = temp

a = tempC['site_no'].unique()
b = tempQ['site_no'].unique()
len(np.intersect1d(a, b))
