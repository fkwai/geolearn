
"""wrap up data for the whole CONUS
some spectial sites:
'02465000' '08068450' '07311600'
"""

from hydroDL.data import usgs
import time
import os
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET
from hydroDL.app import waterQuality
import importlib

# list of site
startDate = pd.datetime(1979, 1, 1)
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoSel')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
rho = 365
nFill = 5

# select referenced basins
tabSel = gageII.readData(
    varLst=['CLASS', 'ROADS_KM_SQ_KM'], siteNoLst=siteNoLstAll)
tabSel = gageII.updateCode(tabSel)
siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()
siteNoLst = siteNoLst[:5]

# caseName = 'refBasins'
caseName = 'temp'

varC = usgs.lstCodeSample
varG = gageII.lstWaterQuality

waterQuality.wrapData(
    caseName, siteNoLst, rho=rho,
    nFill=nFill, varC=varC, varG=varG, targetQ=False)
