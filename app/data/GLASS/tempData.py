
# sherlock can not instll pyhdf. Save temp data instead

from pyhdf.SD import SD, SDC
import glob
import os
import numpy as np
import time
from joblib import Parallel, delayed

yrLst = list(range(1982, 2019))
dLst = list(range(1, 366, 8))
[j1, j2] = [800, 1300]
[i1, i2] = [1100, 2300]


def func(yr):
    tempDir = r'D:\data\GLASS\temp'
    out = np.full([j2-j1, i2-i1, len(dLst)], np.nan)
    varLst = ['LAI', 'FAPAR', 'NPP']
    for var in varLst:
        folder = r'D:\data\GLASS\{}\AVHRR\{}'.format(var, yr)
        saveFile = os.path.join(tempDir, '{}_{}'.format(var, yr))
        if os.path.exists(saveFile+'.npz'):
            continue
        t0 = time.time()
        for iD, d in enumerate(dLst):
            name = '*.V40.A{}{:03d}.*.hdf'.format(yr, d)
            fileLst = glob.glob(os.path.join(folder, name))
            if len(fileLst) == 1:
                hdf = SD(os.path.join(folder, fileLst[0]), SDC.READ)
                data = hdf.select(var)[j1:j2, i1:i2]
                out[:, :, iD] = data
            elif len(fileLst) > 1:
                raise Exception('mutiple of such file')
            else:
                print('no data {} {} {}'.format(var, yr, d))
        print('{} {} {:.2f}'.format(var, yr, time.time()-t0))
        saveFile = os.path.join(tempDir, '{}_{}'.format(var, yr))
        np.savez_compressed(saveFile, out=out)



results = Parallel(n_jobs=-1)(delayed(func)(yr) for yr in yrLst)
