from hydroDL.data import dbBasin, gridMET
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

fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dataName = 'Q90N'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=list(), varF=gridMET.varLst)

fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dataName = 'Q90Nref'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=list(), varF=gridMET.varLst)
