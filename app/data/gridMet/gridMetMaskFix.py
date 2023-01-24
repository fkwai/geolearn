from shapely.geometry import Point, shape
import importlib
import numpy as np
import pandas as pd

import shapefile
from shapely.geometry import shape, Polygon, box
import os
import time
import argparse

from hydroDL.data import gridMET
from hydroDL.utils import gis
from hydroDL import kPath

ncFile = os.path.join(kPath.dirData, 'gridMET', 'etr_1979.nc')
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
maskFolder = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

# find out error sites - too small in mask generation
siteNoLst = [f[:-4]
             for f in sorted(os.listdir(maskFolder)) if f[-3:] == 'npy']
# errLst = list()
# t0 = time.time()
# for k, siteNo in enumerate(siteNoLst):
#     mask = np.load(os.path.join(maskFolder, siteNo+'.npy'))
#     if np.sum(mask) == 0:
#         errLst.append(siteNo)
#     print('\t site {}/{} time cost {:.2f}'.format(
#         k, len(siteNoLst), time.time()-t0), end='\r')
# dfErr = pd.DataFrame(data=errLst)
# dfErr.to_csv(os.path.join(kPath.dirData, 'USGS', 'gridMET',
#                           'errLst'), header=False, index=False)
errLst = pd.read_csv(os.path.join(kPath.dirData, 'USGS',
                                  'gridMET', 'errLst'), dtype=str).values.flatten().tolist()

# fix mask
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
errShpLst = [shp for (shp, rec) in zip(shapeLst, recLst) if rec[2] in errLst]

t, lat, lon = gridMET.readNcInfo(ncFile)
data = gridMET.readNcData(ncFile)
nanMaskAll = np.isnan(data)
nanMask = nanMaskAll.any(axis=0)

k = 1
# siteNo = errLst[k]
# geog = shape(errShpLst[k])
siteNo = siteNoLst[k]
geog = shape(shapeLst[k])
mask = np.zeros([len(lat), len(lon)])
bb = geog.bounds
xx = (bb[0]+bb[2])/2
yy = (bb[1]+bb[3])/2
indX = np.argmin(np.abs(lon - xx))
indY = np.argmin(np.abs(lat - yy))
mask[indY, indX] = 1
if nanMask[indY, indX]:
    print('error!')


t0 = time.time()
mask1 = gis.gridMask(lat, lon, geog, ns=4)
print(time.time()-t0)


t0 = time.time()
mask = np.zeros([len(lat), len(lon)])
indX1 = np.where(lon < bb[0])[0][-1]
indX2 = np.where(lon > bb[2])[0][0]
indY1 = np.where(lat > bb[3])[0][-1]
indY2 = np.where(lat < bb[1])[0][0]
dx = (lon[indX2] - lon[indX1]) / (indX2 - indX1)
dy = (lat[indY1] - lat[indY2]) / (indY2 - indY1)
area = dx*dy
ns = 4
polygon = geog
for i in range(indX1, indX2 + 1):
    for j in range(indY1, indY2 + 1):
        x1 = lon[i] - dx / 2
        x2 = lon[i] + dx / 2
        y1 = lat[j] + dy / 2
        y2 = lat[j] - dy / 2
        # pp = Polygon([(x1, y1), (x1, y2), (x2, y1)])
        pp = box(x1, y2, x2, y1)
        if polygon.contains(pp):
            mask[j, i] = 1
        elif not polygon.intersects(pp):
            mask[j, i] = 0
        else:
            mask[j, i] = polygon.intersection(pp).area/dx/dy
print(time.time()-t0)

import matplotlib.pyplot as plt
aa=mask[indY1:indY2+1,indX1:indX2+1]
plt.imshow(aa)
plt.show()
