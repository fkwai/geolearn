from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]

# wrap up data
caseName = 'HBN-30d'
if waterQuality.exist(caseName):
    wqData = waterQuality.DataModelWQ(caseName)
else:
    wqData = waterQuality.DataModelWQ.new(caseName, siteNoHBN, rho=30)
    ind1 = wqData.indByRatio(0.8)
    ind2 = wqData.indByRatio(0.2, first=False)
    wqData.saveSubset(['first80', 'last20'], [ind1, ind2])

basins.trainModelTS('HBN', 'first80', batchSize=[None, 500], saveName='HBN_opt1', optQ=1)

a, b = [200, None]
