from hydroDL.utils import gis
from shapely.geometry import shape
import shapefile
import time
from hydroDL import kPath
from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from joblib import Parallel, delayed
import json

# only selected sites
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# global to US
[j1, j2] = [800, 1300]
[i1, i2] = [1100, 2300]

# read all masks - save to temp npz files
ns = len(siteNoLst)
maskDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
tempDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'temp')

# load masks
# maskAry = np.ndarray([j2-j1, i2-i1, ns])
# t0 = time.time()
# for k, siteNo in enumerate(siteNoLst):
#     maskFile = os.path.join(maskDir, siteNo)
#     mask = np.load(maskFile+'.npz')['mask']
#     maskAry[:, :, k] = mask[j1:j2, i1:i2]
#     print('{}/{} {:.2f}'.format(k, ns, time.time()-t0))
# tempFile = os.path.join(tempDir, 'mask_Y30N5')
# np.savez_compressed(tempFile, mask=maskAry, siteNoLst=siteNoLst)
tempFile = os.path.join(tempDir, 'mask_Y30N5.npz')
npz = np.load(tempFile)
maskAry = npz['mask']

# construct time
tLst, yrLst, dLst = [list(), list(), list()]
for yr in np.arange(1982, 2019):
    for d in np.arange(1, 366, 8):
        t = np.datetime64(str(yr))+np.timedelta64(d-1, 'D')
        tLst.append(t)
        yrLst.append(yr)
        dLst.append(d)
tAry = np.array(tLst)
nt = len(tAry)

# extract data
varLst = ['LAI', 'FAPAR', 'NPP']
# tempDir = os.path.join(kPath.dirData, 'GLASS', 'temp')
tempDir = r'D:\data\GLASS\temp'
matAll = np.ndarray([len(siteNoLst), nt, len(varLst)])
t0 = time.time()
for yr in np.arange(1982, 2019):
    for iV, var in enumerate(varLst):
        tempFile = os.path.join(tempDir, '{}_{}.npz'.format(var, yr))
        dataTemp = np.load(tempFile)['out']
        t1 = time.time()
        for iD, d in enumerate(np.arange(1, 366, 8)):
            t = np.datetime64(str(yr))+np.timedelta64(d-1, 'D')
            iT = np.where(tAry == t)[0][0]
            data = dataTemp[:, :, iD]
            iy, ix = np.where(data == 2550)
            maskTemp = maskAry.copy()
            maskTemp[iy, ix, :] = 0
            out = np.sum(data[:, :, None]*maskTemp, axis=(0, 1)
                         ) / np.sum(maskTemp, axis=(0, 1))
            matAll[:, iT, iV] = out
        t2 = time.time()
        print('{} {} {:.2f}'.format(yr, var,  time.time()-t0), flush=True)

# save output
outDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'output')
for k, siteNo in enumerate(siteNoLst):
    df = pd.DataFrame(columns=varLst, data=matAll[k, :, :], index=tAry)
    df.index.name = 'date'
    df.to_csv(os.path.join(outDir, siteNo))
    print('saving {}'.format(siteNo), flush=True)
# some issues with the output data. Memory issue during running?