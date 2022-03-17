
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


# read all C/Q data and save as csv - improve future efficiency
# concentrations
dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv')
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfC, dfCF = usgs.readSample(
        siteNo, flag=True, csv=False)
    if dfC is not None:
        dfC.to_csv(os.path.join(dirC, siteNo))
        dfCF.to_csv(os.path.join(dirC, siteNo+'_flag'))
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')


# # read all C/Q data and save as csv - improve future efficiency
# # streamflow
# dirQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', 'csv')
# t0 = time.time()
# for i, siteNo in enumerate(siteNoLst):
#     dfQ = usgs.readStreamflow(siteNo)
#     if dfQ is not None:
#         dfQ.to_csv(os.path.join(dirQ, siteNo))
#     print('\t {}/{} {:.2f}'.format(
#         i, len(siteNoLst), time.time()-t0), end='\r')

