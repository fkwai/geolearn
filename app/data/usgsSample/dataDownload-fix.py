
from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import json

# site inventory
dataFolder = os.path.join(kPath.dirRaw, 'USGS', 'streamflow')
fileSiteNo = os.path.join(kPath.dirUSGS, 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLstAll = dictSite['CONUS']
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
siteNoLstTemp = [f for f in sorted(os.listdir(dataFolder))]
siteNoLst = [f for f in siteNoLstAll if f not in siteNoLstTemp]

# download C/Q data - this will download all elements
errLst = list()
invDir = os.path.join(kPath.dirData, 'USGS', 'inventory')
tabState = pd.read_csv(os.path.join(invDir, 'fips_state_code.csv'))
siteQ = usgs.readUsgsText(os.path.join(invDir, 'inventory_NWIS_streamflow'))
# siteC = usgs.readUsgsText(os.path.join(invDir, 'inventory_NWIS_sample'))

t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    try:
        stateCd = siteQ['state_cd'].loc[siteQ['site_no'] == siteNo].values[0]
        state = tabState['short'].loc[tabState['code'] == int(stateCd)].values[0]
        saveFile = os.path.join(kPath.dirRaw, 'USGS', 'streamflow', siteNo)
        if not os.path.exists(saveFile):
            usgs.downloadDaily(siteNo, ['00060'], state, saveFile)
        # saveFile = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
        # if not os.path.exists(saveFile):
        #     usgs.downloadSample(siteNo, state, saveFile)
    except:
        errLst.append(siteNo)
    tc = time.time()-t0
    print('\t site {}/{} time cost {:.2f}'.format(k, len(siteNoLst), tc))
# df = pd.DataFrame(errLst)
# saveFile = os.path.join(kPath.dirData, 'USGS', 'errLst.csv')
# df.to_csv(saveFile, index=False, header=False)
# NO ERROR SITES!!

# a = siteQ['site_no'].tolist()
# b = siteNoLstAll
# c = [f for f in b if f not in a]