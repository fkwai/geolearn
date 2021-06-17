
from hydroDL.data import dbBasin
import json
import os
from hydroDL import kPath
import numpy as np

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'G400'
dictSiteName = 'dict{}.json'.format(dataName)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['rmTK']
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)

# subset pick by random
DF = dbBasin.DataFrameBasin(dataName)
t1 = dbBasin.func.pickRandT(DF.t, 0.2)
t2 = dbBasin.func.pickRandT(DF.t, 0.2, pick=False)
DF.createSubset('pkRT20', dateLst=t1)
DF.createSubset('rmRT20', dateLst=t2)

# normalization
DFN = dbBasin.func.localNorm(DF, subset='rmRT20')
DFN.saveAs(dataName+'Norm')

