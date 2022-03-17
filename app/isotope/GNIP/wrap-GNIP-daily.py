# Global Sinusoidal Seasonality

from hydroDL import kPath, utils
import pandas as pd
import os
from hydroDL.data import GNIP
import importlib
importlib.reload(GNIP)

dirGNIP = os.path.join(kPath.dirData, 'isotope', 'GNIP')
fileD = os.path.join(dirGNIP, 'GNIP-Daily.csv')
dfD = pd.read_csv(fileD)
dfD = dfD.rename(columns=GNIP.rnCol)

siteNoLst = list(set(dfD['WMO']))

# check if rows for one site are identical - all identical
for siteNo in siteNoLst:
    for col in GNIP.siteCol:
        n = len(dfD[dfD['WMO'] == siteNo][col].unique())
        if n != 1:
            print(siteNo, col)
            dfD[dfD['WMO'] == siteNo][col].unique()

# found that there are two 7121502
# rename OSOYOOS EAST to 7121500
for site, siteNo in GNIP.rnSite.items():
    dfD.at[dfD['Site'] == site, 'WMO'] = siteNo

# dfD[dfD['Site'] == 'OSOYOOS EAST']['WMO']
# dfD[dfD['Site'] == 'OSOYOOS WEST']['WMO']

# summarize sites
dfS = dfD[GNIP.siteCol].drop_duplicates().set_index('WMO')
dfCount = dfD.groupby('WMO').count()
dfS['O18-count'] = dfCount['O18']
dfS['H2-count'] = dfCount['H2']
dfS['H3-count'] = dfCount['H3']
dfS.to_csv(os.path.join(dirGNIP, 'siteD.csv'))

# save each site to a single file
dirD = os.path.join(dirGNIP, 'GNIP-Daily')
if not os.path.exists(dirD):
    os.mkdir(dirD)
for siteNo in siteNoLst:
    dfV = dfD[dfD['WMO'] == siteNo][GNIP.obsColD]
    dfV[['Date', 'sd', 'ed']] = dfV[['Date', 'sd', 'ed']].apply(pd.to_datetime)
    dfV.set_index('Date').to_csv(os.path.join(dirD, '{}.csv'.format(siteNo)))
