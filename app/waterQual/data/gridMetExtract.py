import importlib
from hydroDL.data import gridMET
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse

workDir = kPath.dirWQ
dataFolder = os.path.join(kPath.dirData, 'gridMET')
maskFolder = os.path.join(kPath.dirData, 'USGS-mask')
saveFolder = os.path.join(kPath.dirData, 'USGS-gridMET', 'raw')


# varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

"""

srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var pr

"""


def extractData(var):
    t0 = time.time()
    file = os.path.join(maskFolder, 'maskAll.npy')
    maskAll = np.load(file)
    print('load all mask time {}'.format(time.time()-t0))

    for yr in range(1979, 2020):
        t0 = time.time()
        ncFile = os.path.join(dataFolder, '{}_{}.nc'.format(var, yr))
        data = gridMET.readNcData(ncFile)
        print('read year {} time {}'.format(yr, time.time()-t0))

        nt, ny, nx = data.shape
        ny, nx, ns = maskAll.shape
        data[np.isnan(data)] = 0

        m1 = data.reshape(nt, -1)
        m2 = maskAll.reshape(-1, ns)
        out = np.matmul(m1, m2)
        file = os.path.join(saveFolder, '{}_{}'.format(var, yr))
        np.save(file, out)
        print('finished year {} time {}'.format(yr, time.time()-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-var', dest='var', type=str)
    # parser.add_argument('-yr', dest='yr', type=int)
    args = parser.parse_args()
    extractData(args.var)


""" save all mask to one file

ncFile = os.path.join(dataFolder, 'pr_1979.nc')
data = gridMET.readNcData(ncFile)
nanMaskAll = np.isnan(data)
nanMask = nanMaskAll.any(axis=0)

fileSiteNo = os.path.join(workDir, 'modelUsgs2', 'siteNoSel')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
maskLst = list()
k = 0
for siteNo in siteNoLst:
    t0 = time.time()
    mask = np.load(os.path.join(maskFolder, siteNo+'.npy'))
    mask[nanMask] = 0
    mask = mask/np.sum(mask)
    maskLst.append(mask)
    print('{} {} time {}'.format(k, siteNo, time.time()-t0))
    k = k+1
maskAll = np.stack(maskLst, axis=2)
file = os.path.join(maskFolder, 'maskAll')
np.save(file, maskAll)

## get nan sites (too small no mask)
# aa=np.sum(maskAll.reshape(-1,ns),axis=0)
# for k in range(ns):
#     print(k,aa[k])
# for kk in list(np.where(np.isnan(aa))[0]):
#     print(siteNoLst[kk])
"""
