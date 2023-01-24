import importlib
import hydroDL.data.gridMET.io as io
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse

# varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

"""
extract gridMet data for basins in gageII
srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var pr

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-var', dest='var', type=str, default='pr')
    parser.add_argument('-syr', dest='syr', type=int, default=1979)
    parser.add_argument('-eyr', dest='eyr', type=int, default=1981)
    parser.add_argument('-smask', dest='smask', type=int, default=0)
    parser.add_argument('-emask', dest='emask', type=int, default=100)
    args = parser.parse_args()
    var = args.var
    syr = args.syr
    eyr = args.eyr
    smask = args.smask
    emask = args.emask

dataFolder = os.path.join(kPath.dirRaw, 'gridMet')
maskFolder = os.path.join(kPath.dirUsgs, 'gridMet', 'mask')
rawFolder = os.path.join(kPath.dirUsgs, 'gridMet', 'raw')

siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteNoLstAll=dfSite['siteNo'].tolist()
siteNoLst = siteNoLstAll[smask:emask]

# not necessary
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


for yr in range(syr, eyr):
    t0 = time.time()
    ncFile = os.path.join(dataFolder, '{}_{}.nc'.format(var, yr))
    data,(t, lat, lon) = io.readNcData(ncFile)
    t, lat, lon = io.readNcInfo(ncFile)
    print('read year {} time {}'.format(yr, time.time()-t0))

    if yr == syr:
        maskAll = readMask(siteNoLst, data)
    ny, nx,nt = data.shape
    ny, nx, ns = maskAll.shape
    data[np.isnan(data)] = 0

    m1 = data.reshape(-1,nt)
    m2 = maskAll.reshape(-1, ns)
    out = np.matmul(m1.T, m2)
    df = pd.DataFrame(data=out, index=t, columns=siteNoLst)
    fileName = os.path.join(
        rawFolder, '{}_{}_{}_{}.csv'.format(var, yr, smask, emask))
    df.to_csv(fileName)
    print('finished year {} time {}'.format(yr, time.time()-t0))
