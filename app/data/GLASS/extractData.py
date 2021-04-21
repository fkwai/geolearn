import time
from hydroDL import kPath
import os
import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=0)
    parser.add_argument('-E', dest='iEnd', type=int, default=3)
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
        t = np.datetime64(str(yr))+np.timedelta64(int(d-1), 'D')
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
    siteNoLst = siteNoLst+list(npz['siteNoLst'])
    print('read mask {} {}'.format(k1, k2), flush=True)
maskAry = np.concatenate(maskLst, axis=2)

# extract data
varLst = ['LAI', 'FAPAR', 'NPP']
tempDir = os.path.join(kPath.dirData, 'GLASS', 'temp')
# tempDir = r'D:\data\GLASS\temp'
matAll = np.ndarray([len(siteNoLst), nt, len(varLst)])
t0 = time.time()
for yr in np.arange(1982, 2019):
    for iV, var in enumerate(varLst):
        tempFile = os.path.join(tempDir, '{}_{}.npz'.format(var, yr))
        dataTemp = np.load(tempFile)['out']
        t1 = time.time()
        for iD, d in enumerate(np.arange(1, 366, 8)):
            t = np.datetime64(str(yr))+np.timedelta64(int(d-1), 'D')
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
