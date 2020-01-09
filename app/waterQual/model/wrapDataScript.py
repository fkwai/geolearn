
"""wrap up data for the whole CONUS
"""

import os
import pandas as pd
import numpy as np
from hydroDL import kPath
from hydroDL.data import usgs, gageII
import importlib

# list of site
codeLstC = \
    ['00915', '00925', '00930', '00935', '00955', '00940', '00945'] +\
    ['00418', '00419', '39086', '39087'] +\
    ['00301', '00300', '00618', '00681', '00653'] +\
    ['00010', '00530', '00094'] +\
    ['00403', '00408']
startDate = pd.datetime(1979, 1, 1)
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoSel')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# gageII
varCLst = ['NWIS_DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'GEOL_REEDBUSH_DOM', 'GEOL_HUNT_DOM_CODE', 'STREAMS_KM_SQ_KM',
           'BFI_AVE', 'CONTACT', 'FORESTNLCD06', 'PLANTNLCD06',
           'NUTR_BAS_DOM', 'ECO3_BAS_DOM', 'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']
tabC = gageII.readData(varLst=varCLst, siteNoLst=siteNoLst)
tabC=gageII.updateCode(tabC)

tabC = gageII.readData(varLst=['CLASS','NWIS_DRAIN_SQKM'], siteNoLst=siteNoLst)
tabC=gageII.updateCode(tabC)

import matplotlib.pyplot as plt
a=tabC['CLASS'].values
b=tabC['NWIS_DRAIN_SQKM'].values
np.sum(a-1)

# read data
siteNo = '02465000'
# sample
fileC = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
dfC = usgs.readUsgsText(fileC, dataType='sample')
dfC = dfC[dfC['date'] >= startDate]
dfC = dfC[['date']+list(set(codeLstC) & set(dfC.columns.tolist()))]
dfC = dfC.set_index('date').dropna(how='all')
dfC = dfC.groupby(level=0).agg(lambda x: x.mean())

# streamflow
siteNo = '08068450'
fileQ = os.path.join(kPath.dirData, 'USGS', 'dailyTS', siteNo)
dfQ = usgs.readUsgsText(fileQ, dataType='streamflow')
dfQ = dfQ[dfQ['date'] >= startDate]
if '00060_00001' in dfQ.columns and '00060_00002' in dfQ.columns:
    # fill nan using other two fields
    avgQ = dfQ[['00060_00001', '00060_00002']].mean(axis=1, skipna=False)
    dfQ['00060_00003'] = dfQ['00060_00003'].fillna(avgQ)
    dfQ = dfQ[['date', '00060_00003']]
else:
    dfQ = dfQ[['date', '00060_00003']]


# forcing
fileF = os.path.join(kPath.dirData, 'USGS', 'gridMet', siteNo)
dfF = pd.read_csv(fileF)


# # fill nan of small gap
# dfQ['00060_00003']=dfQ['00060_00003'].interpolate(limit=5)
# dfQ.to_csv('temp.csv')


# forcing
fileF = os.path.join(kPath.dirData, 'USGS', 'gridMet', siteNo)
dfF = pd.read_csv(fileF)

# merge data

# dfQ[['date', '00060_00003']]
# temp = pd.merge(dfQ, dfC, how='outer', on='date')
