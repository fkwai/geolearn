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
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=0)
    parser.add_argument('-E', dest='iEnd', type=int, default=6)
    args = parser.parse_args()
    iStart = args.iStart
    iEnd = args.iEnd

# default parameters
[j1, j2] = [800, 1300]
[i1, i2] = [1100, 2300]
ns = 7111
maskDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
tempDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'temp')

# from temp masks by tempMask.py
k1Lst = np.arange(0, ns, 100)
k2Lst = np.append(k1Lst[1:], ns)
nk = len(k1Lst)

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


# load mask
maskLst = list()
siteNoLst = list()
for k in range(iStart, iEnd):
    k1 = k1Lst[k]
    k2 = k2Lst[k]
    tempFile = os.path.join(tempDir, 'mask{}_{}'.format(k1, k2))
    npz = np.load(tempFile+'.npz')
    maskLst.append(npz['mask'])
    siteNoLst.append(npz['siteNoLst'])
    print('read mask {} {}'.format(k1, k2), flush=True)
maskAry = np.concatenate(maskLst)

# extract data
varLst = ['LAI', 'FAPAR', 'NPP']
matAll = np.ndarray([len(siteNoLst), nt, len(varLst)])
t0 = time.time()
for iT, t in enumerate(tAry):
    yr = yrLst[iT]
    d = dLst[iT]
    for iV, var in enumerate(varLst):
        if var == 'NPP':
            folder = os.path.join(
                kPath.dirData, 'GLASS', var, 'AVHRR', 'GLASS_NPP_005D', str(yr))
        else:
            folder = os.path.join(
                kPath.dirData, 'GLASS', var, 'AVHRR', str(yr))
        name = '*.V40.A{}{:03d}.*.hdf'.format(yr, d)
        fileLst = glob.glob(os.path.join(folder, name))
        if len(fileLst) == 1:
            hdf = SD(os.path.join(folder, fileLst[0]), SDC.READ)
            data = hdf.select(var)[j1:j2, i1:i2]
        else:
            raise Exception('No / mutiple of such file')

        iy, ix = np.where(data == 2550)
        maskTemp = maskAry.copy()
        maskTemp[iy, ix, :] = 0
        out = np.sum(data[:, :, None]*maskTemp, axis=(0, 1)) / \
            np.sum(maskTemp, axis=(0, 1))
        matAll[:, iT, iV] = out
        print('{} {} {} {:.2f}'.format(t, var, k1, time.time()-t0), flush=True)

# save output
outDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'output')
for k, siteNo in enumerate(siteNoLst):
    df = pd.DataFrame(columns=varLst, data=matAll[k, :, :], index=tAry)
    df.index.name = 'date'
    df.to_csv(os.path.join(outDir, siteNo))
    print('saving {}'.format(siteNo), flush=True)
