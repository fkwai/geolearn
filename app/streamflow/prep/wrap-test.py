
import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath, utils
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import basins
from hydroDL.app import waterQuality

"""
instead of saving time series by rho, save the full time series here. 
f and q will be saved in full matirx
c will saved in sparse matrix 
"""

# load sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

freq = 'D'
nFill = 5
sdStr = '1979-01-01'
edStr = '2019-12-31'
# ts data
varF = gridMET.varLst
varQ = usgs.varQ
varG = gageII.lstWaterQuality
# varC=

# gageII
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
fLst, qLst, gLst = [list() for x in range(3)]

infoLst = list()
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    t1 = time.time()
    varLst = varQ+varF
    df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq=freq)
    # streamflow
    tempQ = pd.DataFrame({'date': tR}).set_index('date').join(df[varQ])
    qLst.append(tempQ.values)
    # forcings
    tempF = pd.DataFrame({'date': tR}).set_index('date').join(df[varF])
    tempF = tempF.interpolate(
        limit=nFill, limit_direction='both', limit_area='inside')
    fLst.append(tempF.values)
    # geog
    gLst.append(tabG.loc[siteNo].values)
    t2 = time.time()
    print('{} on site {} reading {:.3f} total {:.3f}'.format(
        i, siteNo, t2-t1, t2-t0))
f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float32)
q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float32)
g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float32)


# save
caseName = 'Q90'
saveFolder = os.path.join(kPath.dirWQ, 'trainDataFull', caseName)
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
np.save(os.path.join(saveFolder, 'Q'), q)
np.save(os.path.join(saveFolder, 'F'), f)
np.save(os.path.join(saveFolder, 'G'), g)
dictData = dict(name=caseName, varG=varG,  varQ=varQ, varF=varF,
                sd=sdStr, ed=edStr, siteNoLst=siteNoLst)
with open(os.path.join(saveFolder, 'info')+'.json', 'w') as fp:
    json.dump(dictData, fp, indent=4)
