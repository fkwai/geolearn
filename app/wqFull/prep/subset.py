from hydroDL.data import dbBasin, usgs
import numpy as np
import os
import json 
from hydroDL import kPath

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y28N5.json')) as f:
    dictSite = json.load(f)

dataNameLst = ['bsWN5', 'bsDN5', 'brWN5', 'brDN5']
addLst = ['comb', 'rmT', 'rmTK', 'rmTKH']
for dataName in dataNameLst:
    DM = dbBasin.DataModelFull(dataName)
    for code in usgs.newC+addLst:
        siteNoLst = dictSite[code]
        DM.saveSubset(code, siteNoLst)
