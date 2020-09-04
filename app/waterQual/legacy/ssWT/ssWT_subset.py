
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import json
import os

dataName = 'ssWT'
wqData = waterQuality.DataModelWQ(dataName)

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
with open(os.path.join(dirInv, 'dictStableSites_0610_0220.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# create subset
for code in dictSite.keys():
    print(code)
    siteNoLst = dictSite[code]
    b1 = wqData.info['siteNo'].isin(siteNoLst).values
    if code != 'comb':
        b2 = ~np.isnan(wqData.c[-1, :, wqData.varC.index(code)])
        info = wqData.info[b1 & b2]
    else:
        info = wqData.info[b1]
    indY1, indY2 = waterQuality.indYrOddEven(info)
    wqData.saveSubset('{}-Y1'.format(code), indY1)
    wqData.saveSubset('{}-Y2'.format(code), indY2)
