
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

# read site inventory
workDir = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileInvC = os.path.join(workDir, 'inventory_NWIS_sample')
fileInvQ = os.path.join(workDir, 'inventory_NWIS_streamflow')

# look up sample for interested sample sites
# see exploreSample.py for this file
fileCountC = os.path.join(workDir, 'count_NWIS_sample_gageII')
tabC = pd.read_csv(fileCountC, dtype={'site_no': str}).set_index('site_no')
codeLst = waterQuality.codeLst
tabSite = tabC[codeLst]

# download C/Q data - this will download all elements
siteNoLst = tabSite.index.tolist()
errLst = list()
tabState = pd.read_csv(os.path.join(workDir, 'fips_state_code.csv'))
siteQ = usgs.readUsgsText(os.path.join(workDir, 'inventory_NWIS_streamflow'))
t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    try:
        stateCd = siteQ['state_cd'].loc[siteQ['site_no'] == siteNo].values[0]
        state = tabState['short'].loc[tabState['code'] == int(stateCd)].values[0]
        saveFile = os.path.join(kPath.dirData, 'USGS', 'streamflow', siteNo)
        if not os.path.exists(saveFile):
            usgs.downloadDaily(siteNo, ['00060'], state, saveFile)
        saveFile = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
        if not os.path.exists(saveFile):
            usgs.downloadSample(siteNo, state, saveFile)
    except:
        errLst.append(siteNo)
    ns = len(siteNoLst)
    tc = time.time()-t0
    print('\t site {}/{} time cost {:.2f}'.format(k, ns, tc), end='\r')
df = pd.DataFrame(errLst)
saveFile = os.path.join(kPath.dirData, 'USGS', 'errLst.csv')
df.to_csv(saveFile, index=False,header=False)


# # list of site - should not do this! wrap up code in one script
# startDate = pd.datetime(1979, 1, 1)
# fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoSel')
# siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
# rho = 365
# nFill = 5

# # select referenced basins
# tabSel = gageII.readData(
#     varLst=['CLASS'], siteNoLst=siteNoLstAll)
# tabSel = gageII.updateCode(tabSel)
# siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()
# siteNoLst = siteNoLst[:5]

# # caseName = 'refBasins'
# caseName = 'temp'

# varC = usgs.lstCodeSample
# varG = gageII.lstWaterQuality

# waterQuality.wrapData(
#     caseName, siteNoLst, rho=rho,
#     nFill=nFill, varC=varC, varG=varG, targetQ=False)
