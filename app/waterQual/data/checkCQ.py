"""
Look at read C data after 1979-01-01, summarize sample combinations count
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
tabC = pd.read_csv(fileCountC, dtype={'site_no': str})
tabC = tabC.set_index('site_no')
siteNoLst = tabC.index.tolist()
codeLst = tabC.columns.tolist()

# summarize relation between variables
# dictSum = dict()
# t0 = time.time()
# for i, siteNo in enumerate(siteNoLst):
#     dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst,
#                           startDate=pd.datetime(1979, 1, 1))
#     dfC.to_csv(os.path.join(kPath.dirData, 'USGS',
#                             'sample', 'csv', siteNo+'.csv'))
#     for k, row in dfC.iterrows():
#         temp = dfC.columns[~pd.isna(row)].tolist()
#         dictName = '-'.join(temp)
#         if dictName not in dictSum:
#             dictSum[dictName] = 1
#         else:
#             dictSum[dictName] = dictSum[dictName]+1
#     print('\t {}/{} {:.2f}'.format(
#         i, len(siteNoLst), time.time()-t0), end='\r')

# tab = pd.DataFrame.from_dict(dictSum, orient='index')
# tab = tab.sort_values(0, ascending=False)
# tab.to_csv(os.path.join(dirInv, 'codeCombCount'), header=False)

# read all C/Q data and save as csv - improve future efficiency
dirQ = os.path.join(kPath.dirData, 'USGS', 'streamflow')
dirC = os.path.join(kPath.dirData, 'USGS', 'sample')
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfQ = usgs.readStreamflow(siteNo)
    dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst)
    if (dfQ is not None) and (dfC is not None):
        dfQ.to_csv(os.path.join(dirQ, 'csv', siteNo))
        dfC.to_csv(os.path.join(dirC, 'csv', siteNo))
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')

siteQLst = os.listdir(os.path.join(dirQ, 'csv'))
dfSiteNo = pd.DataFrame(data=sorted(siteQLst))
dfSiteNo.to_csv(os.path.join(dirInv, 'siteNoLst'), index=False, header=False)
