import importlib
from hydroDL.data import gridMET
import numpy as np
import pandas as pd
import os
import time
import argparse
from hydroDL import kPath


workDir = kPath.dirWQ
dataFolder = os.path.join(kPath.dirData,'gridMET')
maskFolder = os.path.join(kPath.dirData,'USGS-mask')
saveFolder = os.path.join(kPath.dirData,'USGS-gridMET','raw')

# varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

""" 

srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetExtract.py -var pr

"""


def runJob(var):
    syr = 1979
    eyr = 2019
    fileSiteNo = os.path.join(workDir, 'modelUsgs2', 'siteNoSel')
    siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

    dataLst = list()
    for yr in range(syr, eyr+1):
        t0 = time.time()
        ncFile = os.path.join(dataFolder, '{}_{}.nc'.format(var, yr))
        data = gridMET.readNcData(ncFile)
        dataLst.append(data)
        print('read year {} time {}'.format(yr, time.time()-t0))
    out = np.concatenate(dataLst, axis=0)

    nt, ny, nx = out.shape
    maskNanAll = np.isnan(out)
    maskNan = maskNanAll.any(axis=0)
    out[maskNanAll] = 0

    k = 0
    for siteNo in siteNoLst:
        t0 = time.time()
        mask = np.load(os.path.join(maskFolder, siteNo+'.npy'))
        mask[maskNan] = 0
        if np.sum(mask) == 0:
            ts = np.full(nt, np.nan)
        else:
            ts = np.matmul(out.reshape(nt, -1), mask.reshape(-1))/np.sum(mask)
        file = os.path.join(saveFolder, siteNo+'_'+var)
        np.save(file, ts)
        k = k+1
        print('{} saved site {} time {}'.format(k, siteNo, time.time()-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-var', dest='var', type=str)
    args = parser.parse_args()
    runJob(args.var)
