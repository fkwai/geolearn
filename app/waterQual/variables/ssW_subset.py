
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import json
import os

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
with open(os.path.join(dirInv, 'dictStableSites_0610_0220.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# create subset
for code in dictSite.keys():
    print(code)
    siteNoLst = dictSite[code]
    info = wqData.info[wqData.info['siteNo'].isin(siteNoLst)]
    indY1, indY2 = waterQuality.indYrOddEven(info)
    wqData.saveSubset('{}-Y1'.format(code), indY1)
    wqData.saveSubset('{}-Y2'.format(code), indY2)


varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']

varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
