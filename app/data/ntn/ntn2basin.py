from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, ntn
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import importlib

dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')


importlib.reload(ntn)
crdNTN = ntn.loadSite()
crdUSGS = ntn.loadCrdUSGS()
ntnIdLst = crdNTN.index.tolist()
usgsIdLst = crdUSGS.index.tolist()
# freq = 'W'
freq = 'D'

# read all ntn
dictNTN = dict()
for k, ntnId in enumerate(ntnIdLst):
    print(k, ntnId)
    tab = ntn.readSite(ntnId, freq)
    dictNTN[ntnId] = tab

# assign nearest to USGS
distMax = 300
t = tab.index
if freq == 'D':
    usgsFolder = os.path.join(dirNTN, 'usgs', 'daily')
elif freq == 'W':
    usgsFolder = os.path.join(dirNTN, 'usgs', 'weekly')

distMax = 300
t0 = time.time()
for k, usgsId in enumerate(usgsIdLst):
    # usgsId = '04024098'

    x = crdUSGS.loc[usgsId]['x']
    y = crdUSGS.loc[usgsId]['y']
    dist = np.sqrt((x-crdNTN['x'])**2+(y-crdNTN['y'])**2)
    dist = dist.drop(dist[dist > distMax*1000].index)
    data = np.full([len(t), len(ntn.varLst)], np.nan)
    distOut = np.full(len(t), np.nan)
    idOut = np.full(len(t), np.nan, dtype=object)
    while len(dist) > 0:
        ntnId = dist.idxmin()
        temp = dictNTN[ntnId].values
        matNan = np.isnan(data)
        indRow = np.unique(np.where(matNan)[0])
        data[matNan] = temp[matNan]
        idOut[indRow] = ntnId
        distOut[indRow] = dist[ntnId]
        dist = dist.drop(ntnId)
        indRow = np.unique(np.where(np.isnan(data))[0])
        if len(indRow) == 0:
            break
        # end of while
    distOut[indRow] = np.nan
    idOut[indRow] = np.nan
    df = pd.DataFrame(index=t, columns=ntn.varLst, data=data)
    df['distNTN'] = distOut
    df['idNTN'] = idOut
    df.index.name = 'date'
    df.to_csv(os.path.join(usgsFolder, usgsId))
    print('{} {} {:.3f}'.format(k, usgsId, time.time()-t0))
