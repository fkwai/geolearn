
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import pandas as pd
import json
import os


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)

# dataNameLst = ['sbW', 'sbWT', 'sbD', 'sbDT']
dataNameLst = ['nbW', 'nbWT']
for dataName in dataNameLst:
    wqData = waterQuality.DataModelWQ(dataName)
    info = wqData.info
    info['yr'] = pd.DatetimeIndex(info['date']).year
    for code in usgs.newC+['comb']:
        print(dataName, code)
        siteNoLst = dictSite[code]
        bs = info['siteNo'].isin(siteNoLst)
        b1 = (info['yr'] <= 2016).values
        b2 = (info['yr'] > 2016).values
        if code == 'comb':
            ind1 = info.index[b1 & bs].values
            ind2 = info.index[b2 & bs].values
        else:
            if len(wqData.c.shape) == 2:
                bv = ~np.isnan(wqData.c[:, wqData.varC.index(code)])
            elif len(wqData.c.shape) == 3:
                bv = ~np.isnan(wqData.c[-1, :, wqData.varC.index(code)])
            ind1 = info.index[b1 & bs & bv].values
            ind2 = info.index[b2 & bs & bv].values
        wqData.saveSubset('{}-B16'.format(code), ind1)
        wqData.saveSubset('{}-A16'.format(code), ind2)
