from hydroDL.data import dbBasin, gageII
import os
import pandas as pd
from hydroDL import kPath, utils

"""
instead of saving time series by rho, save the full time series here.
f and q will be saved in full matirx
c will saved in sparse matrix
"""

# load sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dataName = 'QN90ref'
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, varG=gageII.varLstEx)
DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')

fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dataName = 'QN90'
DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, varG=gageII.varLstEx)
DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')