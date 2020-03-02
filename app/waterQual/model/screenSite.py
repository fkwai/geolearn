from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import os

# list of site - generate from checkCQ.py, 5978 sites in total
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# read all data for screening


fileCountC = os.path.join(kPath.dirData, 'USGS',
                          'inventory', 'count_NWIS_sample_gageII')
tabC = pd.read_csv(fileCountC, dtype={'site_no': str})
tabC = tabC.set_index('site_no')
siteNoLst = tabC.index.tolist()

# # select referenced basins
tabSel = gageII.readData(
    varLst=['CLASS'], siteNoLst=siteNoLstAll)
tabSel = gageII.updateCode(tabSel)
siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()
siteNoLst = siteNoLst[:5]

# # caseName = 'refBasins'
# caseName = 'temp'

# varC = usgs.lstCodeSample
# varG = gageII.lstWaterQuality

# waterQuality.wrapData(
#     caseName, siteNoLst, rho=rho,
#     nFill=nFill, varC=varC, varG=varG, targetQ=False)
