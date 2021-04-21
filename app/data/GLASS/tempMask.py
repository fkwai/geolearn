from hydroDL.utils import gis
from shapely.geometry import shape
import shapefile
import time
from hydroDL import kPath
from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import numpy as np

# read shapefiles to get siteNoLst
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
sf = shapefile.Reader(shpFile)
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

# global to US
[j1, j2] = [800, 1300]
[i1, i2] = [1100, 2300]

# read all masks - save to temp npz files
ns = len(siteNoLst)
maskDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
tempDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'temp')
k1Lst = np.arange(0, ns, 100)
k2Lst = np.append(k1Lst[1:], ns)
nk = len(k1Lst)
t0 = time.time()
for k1, k2 in zip(k1Lst, k2Lst):
    siteNoTemp = siteNoLst[k1:k2]
    maskAry = np.ndarray([j2-j1, i2-i1, k2-k1])
    print(k1, k2)
    for k, siteNo in enumerate(siteNoTemp):
        maskFile = os.path.join(maskDir, siteNo)
        mask = np.load(maskFile+'.npz')['mask']
        maskAry[:, :, k] = mask[j1:j2, i1:i2]
        print('{}/{} {:.2f}'.format(k, k2-k1, time.time()-t0))
    tempFile = os.path.join(tempDir, 'mask{}_{}'.format(k1, k2))
    np.savez_compressed(tempFile, mask=maskAry, siteNoLst=siteNoTemp)
