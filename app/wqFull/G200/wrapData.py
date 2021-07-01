
from hydroDL.data import dbBasin
import json
import os
from hydroDL import kPath
import numpy as np

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'G200'
dictSiteName = 'dict{}.json'.format(dataName)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['rmTK']
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)
DF = dbBasin.DataFrameBasin(dataName)

# normalization
DFN = dbBasin.func.localNorm(DF, subset='all')
DFN.saveAs(dataName+'N')
