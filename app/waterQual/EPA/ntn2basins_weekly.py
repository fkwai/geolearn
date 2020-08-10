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

# read crds
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
crdNTN = pd.read_csv(os.path.join(dirNTN, 'crdNTN.csv'), index_col='siteid')
crdNTN = crdNTN.drop(['CO83', 'NC30', 'WI19'])
crdUSGS = pd.read_csv(os.path.join(
    dirNTN, 'crdUSGS.csv'), dtype={'STAID': str})
crdUSGS = crdUSGS.set_index('STAID')

#
dfCountYr = pd.read_csv(os.path.join(kPath.dirData, 'USGS',
                                     'inventory', 'siteCountWeekly-Y10.csv'), dtype={'siteNo': str})
dfCountYr = dfCountYr.set_index('siteNo')
codeLst = sorted(usgs.codeLst)
temp = dfCountYr[codeLst[2:]] >= 6
temp.sum(axis=1).value_counts().sort_index(ascending=False).cumsum()
tempSum = temp.sum(axis=1)
siteSel = tempSum.index[tempSum >= 16]
siteNoLst = siteSel.tolist()
crdUSGS = crdUSGS.loc[siteNoLst]
# crdUSGS = crdUSGS[crdUSGS['CLASS'] == 1]

usgsIdLst = crdUSGS.index.tolist()
ntnIdLst = crdNTN.index.tolist()
t = pd.date_range(start='1979-01-01', end='2019-12-31', freq='W-TUE')
t = t[1:]
ntnFolder = os.path.join(dirNTN, 'csv', 'weeklyRaw')
usgsFolder = os.path.join(dirNTN, 'usgs', 'weeklyRaw')

# read ntn
dictNTN = dict()
for k, ntnId in enumerate(ntnIdLst):
    print(k, ntnId)
    df = pd.read_csv(os.path.join(ntnFolder, ntnId), index_col='date')
    df.index = pd.to_datetime(df.index)
    dictNTN[ntnId] = df

# assign nearest to USGS
t0 = time.time()
for k, usgsId in enumerate(usgsIdLst):
    # usgsId = '04024098'
    x = crdUSGS.loc[usgsId]['x']
    y = crdUSGS.loc[usgsId]['y']
    dist = np.sqrt((x-crdNTN['x'])**2+(y-crdNTN['y'])**2)
    dist = dist.drop(dist[dist > 500*1000].index)
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
        # end of while
    distOut[indRow] = np.nan
    idOut[indRow] = np.nan
    df = pd.DataFrame(index=t, columns=ntn.varLst, data=data)
    df['distNTN'] = distOut
    df['idNTN'] = idOut
    df.index.name = 'date'
    df.to_csv(os.path.join(usgsFolder, usgsId))
    print('{} {} {:.3f}'.format(k, usgsId, time.time()-t0))
