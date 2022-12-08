import importlib
from hydroDL.data.cmip import io
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse
import json

mLst = ['MPI-ESM1-2-XR', 'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']
modelName = 'MPI-ESM1-2-XR'


dataFolder = os.path.join(kPath.dirRaw, 'CMIP', modelName)
outFolder = os.path.join(kPath.dirData, 'USGS','CMIP', modelName)
maskFolder = os.path.join(kPath.dirData, 'USGS', 'mask', 'CMIP', modelName)
rawFolder = os.path.join(outFolder, 'raw')
saveFolder = os.path.join(outFolder, 'output')


if not os.path.exists(rawFolder):
    os.makedirs(rawFolder)

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLst = dictSite['CONUS']


df = io.walkFile()
var = 'pr'
lab = 'hist-1950'
yLst = list(range(1980, 2015, 5))
sLst = list(range(0, len(siteNoLst), 2000))
varLst = ['pr', 'tasmax', 'tasmin']
for var in varLst:
    for y1, y2 in zip(yLst[:-1], yLst[1:]):
        d1 = np.datetime64('{}-01-01'.format(y1))
        d2 = np.datetime64('{}-01-01'.format(y2))
        data, lat, lon, t = io.readCMIP(
            dfFile=df, var=var, exp=lab, sd=d1, ed=d2, model=modelName)
        nanMaskAll = np.isnan(data)
        nanMask = nanMaskAll.any(axis=-1)
        for s1, s2 in zip(sLst[:-1], sLst[1:]):
            maskLst = list()
            t0 = time.time()
            for k in range(s1, s2):
                siteNo = siteNoLst[k]
                mask = np.load(os.path.join(
                    maskFolder, siteNo+'.npz'))['arr_0']
                mask[nanMask] = 0
                mask = mask/np.sum(mask)
                maskLst.append(mask)
            maskAll = np.stack(maskLst, axis=-1)
            ny, nx, nt = data.shape
            ny, nx, ns = maskAll.shape
            m1 = data.reshape(-1, nt)
            m2 = maskAll.reshape(-1, ns)
            out = np.matmul(m1.T, m2)
            df = pd.DataFrame(data=out, index=t, columns=siteNoLst[s1:s2])
            tempName = '{}_{}_{}_{}_{}.csv'.format(var, y1, y2, s1, s2)
            fileName = os.path.join(rawFolder, tempName)
            df.to_csv(fileName)
            print('finished {} time {}'.format(tempName, time.time()-t0))

# validate
# out2=np.zeros([nt,ns])
# for j in range(nt):
#     for i in range(ns):
#         out2[j,i]=np.sum(data[:, :, j]*maskAll[:, :, i])
# np.sum(np.abs(out2-out))

