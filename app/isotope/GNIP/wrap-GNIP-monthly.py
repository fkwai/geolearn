# Global Sinusoidal Seasonality
from hydroDL import kPath, utils
import pandas as pd
import os
from hydroDL.data import GNIP
import importlib
importlib.reload(GNIP)

fileM1 = os.path.join(kPath.dirData, 'isotope',
                      'GNIP', 'GNIP-Monthly-westUS.csv')
dfM1 = pd.read_csv(fileM1)
dfM1 = dfM1.rename(columns=GNIP.rnCol)
fileM2 = os.path.join(kPath.dirData, 'isotope',
                      'GNIP', 'GNIP-Monthly-eastUS.csv')
dfM2 = pd.read_csv(fileM2)
dfM2 = dfM2.rename(columns=GNIP.rnCol)


# check if rows for one site are identical - all identical
for dfM in [dfM1, dfM2]:
    siteNoLst = list(set(dfM['WMO']))
    for siteNo in siteNoLst:
        for col in GNIP.siteCol:
            n = len(dfM[dfM['WMO'] == siteNo][col].unique())
            if n != 1:
                print(siteNo, col)
                dfM[dfM['WMO'] == siteNo][col].unique()
# find duplicated sites - {7225600, 7185001}
s1 = set(dfM1['WMO'])
s2 = set(dfM2['WMO'])
s1.intersection(s2)
# check if they are identical - yes
for siteNo in [7225600, 7185001]:
    a = dfM1[dfM1['WMO'] == siteNo].fillna(-999)
    b = dfM2[dfM2['WMO'] == siteNo].fillna(-999)
    n = len(a)
    for col in a.columns:
        for k in range(n):
            bb = a[col].iloc[k] == b[col].iloc[k]
            if not bb:
                print(siteNo, col, k)

dfM = pd.concat([dfM1, dfM2]).drop_duplicates()  # should be find


# summarize sites
dirGNIP = os.path.join(kPath.dirData, 'isotope', 'GNIP')
dfS = dfM[GNIP.siteCol].drop_duplicates().set_index('WMO')
dfCount = dfM.groupby('WMO').count()
dfS['O18-count'] = dfCount['O18']
dfS['H2-count'] = dfCount['H2']
dfS['H3-count'] = dfCount['H3']
dfS.to_csv(os.path.join(dirGNIP, 'siteM.csv'))

# save each site to a single file
dirM = os.path.join(dirGNIP, 'GNIP-Monthly')
if not os.path.exists(dirM):
    os.mkdir(dirM)
for siteNo in siteNoLst:
    dfV = dfM[dfM['WMO'] == siteNo][GNIP.obsColM]
    dfV[['Date', 'sd', 'ed']] = dfV[['Date', 'sd', 'ed']].apply(pd.to_datetime)
    dfV.set_index('Date').to_csv(os.path.join(dirM, '{}.csv'.format(siteNo)))
