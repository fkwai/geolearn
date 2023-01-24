import importlib
import hydroDL.data.gridMET.io as io
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse
import json

dataFolder = os.path.join(kPath.dirRaw, 'gridMET')

gridMetFolder = os.path.join(kPath.dirUSGS, 'gridMET')
maskFolder = os.path.join(gridMetFolder, 'mask')
rawFolder = os.path.join(gridMetFolder, 'raw')
outFolder = os.path.join(gridMetFolder, 'output')
syr = 1979
eyr = 2020

fileSiteNo = os.path.join(kPath.dirUSGS, 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLstAll = dictSite['CONUS']
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
siteNoLstTemp = [f for f in sorted(os.listdir(outFolder))]
siteNoLst = [f for f in siteNoLstAll if f not in siteNoLstTemp]


def readMask(siteNoLst, data):
    nanMaskAll = np.isnan(data)
    nanMask = nanMaskAll.any(axis=-1)
    maskLst = list()
    t0 = time.time()
    for k, siteNo in enumerate(siteNoLst):
        mask = np.load(os.path.join(maskFolder, siteNo+'.npz'))['arr_0']
        mask[nanMask] = 0
        mask = mask/np.sum(mask)
        maskLst.append(mask)
        print('{} {} time {}'.format(k, siteNo, time.time()-t0))
    maskAll = np.stack(maskLst, axis=2)
    return maskAll


for var in varLst:
    for yr in range(syr, eyr):
        t0 = time.time()
        ncFile = os.path.join(dataFolder, '{}_{}.nc'.format(var, yr))
        data, (lat, lon, t) = io.readNcData(ncFile)
        print('read year {} time {}'.format(yr, time.time()-t0))

        if yr == syr and var == varLst[0]:
            maskAll = readMask(siteNoLst, data)
        ny, nx, nt = data.shape
        ny, nx, ns = maskAll.shape
        data[np.isnan(data)] = 0

        m1 = data.reshape(-1, nt)
        m2 = maskAll.reshape(-1, ns)
        out = np.matmul(m1.T, m2)
        df = pd.DataFrame(data=out, index=t, columns=siteNoLst)
        fileName = os.path.join(
            rawFolder, '{}_{}_fix.csv'.format(var, yr))
        df.to_csv(fileName)
        print('finished year {} time {}'.format(yr, time.time()-t0))
