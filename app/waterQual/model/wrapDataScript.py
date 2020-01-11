
"""wrap up data for the whole CONUS
some spectial sites:
'02465000' '08068450'
"""

import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET
import importlib

# list of site
startDate = pd.datetime(1979, 1, 1)
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoSel')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
rho = 365
nanLimit = 5

# select referenced basins
tabSel = gageII.readData(
    varLst=['CLASS', 'ROADS_KM_SQ_KM'], siteNoLst=siteNoLstAll)
tabSel = gageII.updateCode(tabSel)
siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()

caseName = 'refBasins'
varC = usgs.lstCodeSample
varG = gageII.lstWaterQuality

dictCase = dict(caseName=caseName, rho=rho, nanLimit=nanLimit,
                varG=varG, varC=varC, siteNoLst=siteNoLst)

# gageII
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLstAll)
tabG = gageII.updateCode(tabG)
# read dataand  merge to three ndarray
# x=[nT,nP,nX], y=[nP,nY], c=[nP,nC]
xLst = list()
yLst = list()
cLst = list()
infoLst = list()
dictSite = dict()
for i, siteNo in enumerate(siteNoLst):
    t0 = time.time()
    dfC = usgs.readSample(siteNo, codeLst=varC, startDate=startDate)
    dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfF = gridMET.readBasin(siteNo)
    t1 = time.time()
    # nTarget = 0
    # tLst = list()
    for k in range(len(dfC)):
        yt = dfC.index[k]
        if yt-pd.Timedelta(days=rho) < startDate:
            continue
        dfX = pd.DataFrame({'date': pd.date_range(
            yt-pd.Timedelta(days=rho-1), yt)}).set_index('date')
        dfX = dfX.join(dfQ)
        dfX = dfX.join(dfF)
        dfX = dfX.interpolate(limit=nanLimit, limit_direction='both')
        if not dfX.isna().values.any():
            xLst.append(dfX.values)
            yLst.append(dfC.iloc[k].values)
            cLst.append(tabG.loc[siteNo].values)
            infoLst.append(dict(siteNo=siteNo, date=yt))
            # nTarget = nTarget + 1
            # tLst.append(yt)
    t2 = time.time()
    # if nTarget != 0:
    #     dictSite[siteNo] = '{:6d} samples, {:%Y/%m/%d} - {:%Y/%m/%d}'.format(
    #         nTarget, tLst[0], tLst[-1])
    print('{} on site {} reading {:.3} processing {:.3}'.format(
        i, siteNo, t1-t0, t2-t1))
x = np.stack(xLst, axis=-1)
y = np.stack(yLst, axis=-1)
c = np.stack(cLst, axis=-1)
info = pd.DataFrame(infoLst).values

saveFolder = os.path.join(kPath.dirWQ, 'tempData')
saveFile = os.path.join(saveFolder, caseName+'.npz')
np.savez(saveFile, x=x, y=y, c=c, info=info)

with open(os.path.join(saveFolder, caseName+'.json'), 'w') as fp:
    json.dump(dictCase, fp, indent=4)
