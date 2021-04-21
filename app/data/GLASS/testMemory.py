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

ns = 7111
k1Lst = np.arange(0, ns, 100)
k2Lst = np.append(k1Lst[1:], ns)
tempDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'temp')

# load mask
t0 = time.time()
mLst = list()
for k1, k2 in zip(k1Lst, k2Lst):
    tempFile = os.path.join(tempDir, 'mask{}_{}'.format(k1, k2))
    npz = np.load(tempFile+'.npz')
    maskAry = npz['mask']
    mLst.append(maskAry)
    print('{} {} {:.2f}'.format(k1,k2, time.time()-t0))
aa=np.concatenate(mLst[:10],axis=2)