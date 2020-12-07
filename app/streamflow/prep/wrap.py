from hydroDL.data import dbBasin
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
# fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

dataName = 'Q90'
dm = dbBasin.DataModelFull.new(dataName, siteNoLst)
