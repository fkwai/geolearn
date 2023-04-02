import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import utils
import numpy as np
import json
from osgeo import ogr, gdal
import matplotlib.gridspec as gridspec

# land cover
fileName = '/mnt/sda/dataRaw/NLCD/2016/nlcd_2016.tif'
matLC = np.zeros([len(lat), 9])
# fileName=os.path.join(kPath.dirVeg, 'const', dictConst['clay'])
ds = gdal.Open(fileName)
gt = ds.GetGeoTransform()
data = ds.GetRasterBand(1).ReadAsArray()
pxA = ((lon - gt[0]) / gt[1]).astype(int)  # x pixel
pyA = ((lat - gt[3]) / gt[5]).astype(int)  # y pixel
for kk, (px, py) in enumerate(zip(pxA, pyA)):
    n = 5
    temp = data[px - n : px + n + 1, py - n : py + n + 1]
    temp = np.floor(temp / 10)
    v = np.zeros(9)
    for k in range(9):
        v[k] = np.sum(temp == k + 1)
    v = v / (n * 2 + 1) ** 2
    matLC[kk, :] = v
lcFile = os.path.join(kPath.dirVeg, 'const', 'nlcd2016-single.npz')
np.savez(
    lcFile,
    mat=matLC,
    lat=lat,
    lon=lon,
    var=np.array(['lc' + str(x) for x in range(1, 10)]),
)