"""
create index of all USGS sites
0. description of all sites
1. a table of siteNo, variable and count
2. same table, after 1979
3. bool matrix if variable observed, 20 variables, and Q, and flag
"""

import os
import fnmatch
from hydroDL import kPath
from hydroDL.data import gageII, usgs
import numpy as np
import pandas as pd
import time


# 0. create inventory of all sites;
folderC = os.path.join((kPath.dirUsgs), 'sample', 'csvAll')
folderQ = os.path.join((kPath.dirUsgs), 'streamflow', 'csv')
sC = [f for f in os.listdir(folderC) if not fnmatch.fnmatch(f, '*_flag')]
sQ = [f for f in os.listdir(folderQ)]
siteNoLst = sorted(set(sC).intersection(set(sQ)))
fileG = os.path.join(kPath.dirData, 'USGS', 'inventory', 'inv-gageII')
tabSite = usgs.read.readUsgsText(fileG)
changeNameDict = {
    'site_no': 'siteNo',
    'station_nm': 'name',
    'dec_lat_va': 'lat',
    'dec_long_va': 'lon',
    'drain_area_va': 'area',
    'huc_cd': 'huc',
}
tabSite = tabSite.rename(columns=changeNameDict)
tabSite.set_index('siteNo', inplace=True)
selectedCol = ['name', 'lat', 'lon', 'area', 'huc']
tabOut = tabSite.loc[siteNoLst, selectedCol]
tabOut.to_csv(os.path.join(kPath.dirUsgs, 'index', 'siteGageII.csv'))

# load all sites
tabOut = pd.read_csv(
    os.path.join(kPath.dirUsgs, 'index', 'siteGageII.csv'), dtype={'siteNo': str}
)
siteNoLst = tabOut['siteNo'].tolist()
t0 = time.time()
# load all data
dictC = dict()
dictCF = dict()
for k, siteNo in enumerate(siteNoLst):
    print('water quality', k, siteNo, time.time() - t0)
    dfC, dfCF = usgs.readSample(siteNo, flag=2)
    dictC[siteNo] = dfC
    dictCF[siteNo] = dfCF
dictQ = dict()
sd = np.datetime64('1979-01-01')
for k, siteNo in enumerate(siteNoLst):
    print('streamflow', k, siteNo, time.time() - t0)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    dictQ[siteNo] = dfQ

# 1. a table of siteNo, variable and count
# 2. same table, after 1979
cLst_all, cLst_a79, cLst_v20, cLst_a79_v20 = [], [], [], []
codeLst = usgs.varC
t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo, time.time() - t0)
    tabC = dictC[siteNo].isna().copy()
    tabC_a79 = tabC[tabC.index >= np.datetime64('1979-01-01')].copy()
    common_col = list(set(tabC.columns).intersection(set(codeLst)))
    tabC_v20 = tabC[common_col].copy()
    tabC_a79_v20 = tabC_a79[common_col].copy()
    for cLst, tab in zip(
        [cLst_all, cLst_a79, cLst_v20, cLst_a79_v20],
        [tabC, tabC_a79, tabC_v20, tabC_a79_v20],
    ):
        tab['obs'] = tab.any(axis=1)
        cs = tab.sum(axis=0)
        cs.name = siteNo
        cLst.append(cs)


# save all
fnLst = ['sampleCount_all', 'sampleCount_a79', 'sampleCount_v20', 'sampleCount_a79_v20']
for cLst, fn in zip([cLst_all, cLst_a79, cLst_v20, cLst_a79_v20], fnLst):
    tab = pd.concat(cLst, axis=1, ignore_index=False).T
    tab = tab.fillna(0).astype(int)
    tab.index.name = 'siteNo'
    sorted_cols = sorted(tab.columns)    
    tab[sorted_cols].to_csv(os.path.join(kPath.dirUsgs, 'index', fn + '.csv'))


# 3. bool matrix if variable observed, 20 variables, and Q, and flag
sdStr = '1979-01-01'
edStr = '2022-12-31'
codeLst = usgs.varC
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
matC = np.ndarray([len(siteNoLst), len(tR), len(codeLst)], dtype=bool)
matQ = np.ndarray([len(siteNoLst), len(tR)], dtype=bool)
matCF = np.ndarray([len(siteNoLst), len(tR), len(codeLst)], dtype=bool)

t0 = time.time()
tabRef = pd.DataFrame(index=tR, columns=codeLst)
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{} {}'.format(k, len(siteNoLst), time.time() - t0))
    tempC = pd.DataFrame(index=tR, columns=codeLst)
    tempC.update(dictC[siteNo])
    tempCF = pd.DataFrame(index=tR, columns=codeLst)
    tempCF.update(dictCF[siteNo])
    tempQ = pd.DataFrame({'date': tR}).set_index('date').join(dictQ[siteNo])
    matCF[k, :, :] = (tempCF == 1).values
    matC[k, :, :] = ~tempC.isna().values
    matQ[k, :] = ~tempQ.isna().values.flatten()

saveFile = os.path.join(kPath.dirUsgs, 'index', 'bMat_A79_V20')
np.savez_compressed(
    saveFile,
    matCF=matCF,
    matC=matC,
    matQ=matQ,
    tR=tR,
    codeLst=codeLst,
    siteNoLst=siteNoLst,
)
