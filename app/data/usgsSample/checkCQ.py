"""
read all C/Q data (after 1979 or all), write csv, sum up siteNoLst

"""

from hydroDL.data import usgs, gageII
from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os

dirUSGS = os.path.join(kPath.dirData, 'USGS')
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')

fileCountC = os.path.join(dirInv, 'count_NWIS_sample_gageII')
# fileCountC = os.path.join(dirInv, 'count_NWIS_sample_all')

tabC = pd.read_csv(fileCountC, dtype={'site_no': str})
tabC = tabC.set_index('site_no')
siteNoLst = tabC.index.tolist()
codeLst = tabC.columns.tolist()

# read all C/Q data and save as csv - improve future efficiency
dirQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', 'csv')
dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv')
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfQ = usgs.readStreamflow(siteNo)
    dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst)
    if (dfQ is not None) and (dfC is not None):
        dfQ.to_csv(os.path.join(dirQ, siteNo))
        dfC.to_csv(os.path.join(dirC, siteNo))
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')


# read all C/Q data after 1979/01/01, regenerate siteNoLst
t0 = time.time()
tempLst = list()
lenQLst = list()
lenCLst = list()
for i, siteNo in enumerate(siteNoLst):
    try:
        dfQ = pd.read_csv(os.path.join(dirQ, siteNo)).set_index('date')
        dfC = pd.read_csv(os.path.join(dirC, siteNo)).set_index('date')
        lenQ = len(dfQ[dfQ.index > '1979-01-01'])
        lenC = len(dfC[dfC.index > '1979-01-01'])
        if (lenQ > 0) & (lenC > 0):
            tempLst.append(siteNo)
            lenQLst.append(lenQ)
            lenCLst.append(lenC)
    except:
        pass
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')
dfSite = pd.DataFrame(dict(siteNo=tempLst, lenQ=lenQLst, lenC=lenCLst))

dfSiteNo = pd.DataFrame(data=sorted(tempLst))
dfSiteNo.to_csv(os.path.join(dirInv, 'siteNoLst-1979'),
                index=False, header=False)
