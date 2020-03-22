from hydroDL import kPath
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.data import gridMET, usgs, gageII
import json

import os
import pandas as pd
import numpy as np

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]


wqData = waterQuality.DataModelWQ('HBN')

varX = usgs.varQ+gridMET.varLst


varF = gridMET.varLst
varG = wqData.varG
varQ = ['00060']
varC = wqData.varC

var = varX[0]

