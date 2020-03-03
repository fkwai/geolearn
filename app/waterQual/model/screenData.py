from hydroDL.data import usgs, gageII, gridMET
from hydroDL import kPath
from hydroDL.app import waterQuality

import pandas as pd
import numpy as np
import os
import time

# list of site - generate from checkCQ.py, 5978 sites in total
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

siteNoLst = siteNoLstAll
caseName = 'basinAll'
waterQuality.wrapData(caseName, siteNoLst)
# dictData, info, q, c, f, g = waterQuality.loadData(caseName)
