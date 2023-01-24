# upgrade code to read flags and save CSV
from hydroDL.data import usgs
from hydroDL import kPath
from hydroDL.app import waterQuality
import os
import pandas as pd



invDir = os.path.join(kPath.dirData, 'USGS', 'inventory')
tabState = pd.read_csv(os.path.join(invDir, 'fips_state_code.csv'))
siteQ = usgs.readUsgsText(os.path.join(
    invDir, 'inventory_NWIS_streamflow'))
siteQ['site_tp_cd']

siteNo = '01017290'

stateCd = siteQ['state_cd'].loc[siteQ['site_no'] == siteNo].values[0]
state = tabState['short'].loc[tabState['code']==int(stateCd)].values[0]
saveFile = os.path.join(kPath.dirRaw, 'USGS', 'streamflow', siteNo)

saveFile = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
usgs.downloadSample(siteNo, state, saveFile)

saveFile = os.path.join(kPath.dirRaw, 'USGS', 'streamflow', siteNo)
usgs.downloadDaily(siteNo, ['00060'], state, saveFile)


dfC = usgs.readSample(siteNo, codeLst=usgs.codeLst, flag=True, csv=False)
