from hydroDL.data import dbBasin
import os
import pandas as pd
from hydroDL import kPath, utils
import json
"""
instead of saving time series by rho, save the full time series here. 
f and q will be saved in full matirx
c will saved in sparse matrix 
"""

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

siteNoLst = dictSite['comb']

dataName = 'sbY30N5'
dm = dbBasin.DataModelFull.new(dataName, siteNoLst)

# dataName = 'sbTest'
# dm = dbBasin.DataModelFull.new(dataName, siteNoLst[:10])
