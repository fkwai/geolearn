import importlib
from hydroDL.data import gridMET
from hydroDL.utils import gis
import numpy as np
import pandas as pd
import os
import time
import argparse

workDir = r'/home/kuaifang/waterQuality/'
dataFolder = r'/home/kuaifang/Data/gridMET/'
maskFolder = r'/home/kuaifang/Data/USGS-mask/'
saveFolder = r'/home/kuaifang/Data/USGS-gridMET/raw/'


# varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

"""

srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var pr

"""

def extractData(var, yr):
    fileSiteNo = os.path.join(workDir, 'modelUsgs2', 'siteNoSel')
    siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

    t0 = time.time()
    ncFile = os.path.join(dataFolder, '{}_{}.nc'.format(var, yr))
    data = gridMET.readNcData(ncFile)
    print('read year {} time {}'.format(yr, time.time()-t0))

    out = data
    nt, ny, nx = out.shape
    nSite = len(siteNoLst)
    maskNanAll = np.isnan(out)
    maskNan = maskNanAll.any(axis=0)
    out[maskNanAll] = 0

    tsMat = np.ndarray([nt, nSite])

    for k in range(nSite):
        t1 = time.time()
        siteNo = siteNoLst[k]
        mask = np.load(os.path.join(maskFolder, siteNo+'.npy'))
        mask[maskNan] = 0
        if np.sum(mask) == 0:
            ts = np.full(nt, np.nan)
        else:
            ts = np.matmul(out.reshape(nt, -1),
                           mask.reshape(-1))/np.sum(mask)
        tsMat[:, k] = ts
        print('\t year {} site {} time {}'.format(
            yr, k, time.time()-t1), end='\r')
    file = os.path.join(saveFolder, '{}_{}'.format(var, yr))
    np.save(file, tsMat)
    print('finished year {} time {}'.format(yr, time.time()-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-var', dest='var', type=str)
    parser.add_argument('-yr', dest='yr', type=int)
    args = parser.parse_args()
    extractData(args.var, args.yr)
