
from hydroDL import kPath
import os
import time
import shapefile
from shapely.geometry import shape
from hydroDL.utils import gis
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=7000)
    parser.add_argument('-E', dest='iEnd', type=int, default=7111)
    parser.add_argument('-R', dest='reMask', type=int, default=True)
    args = parser.parse_args()
    iStart = args.iStart
    iEnd = args.iEnd
    reMask = args.reMask

# read shapefiles
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

# GLASS crd
saveDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
lon = np.arange(-179.975, 180, 0.05)
lat = np.arange(89.975, -90, -0.05)

# remove masked basins from list
if reMask is False:
    maskLst = [f[:-4] for f in os.listdir(saveDir) if f[-3:] == 'npy']
    tempShpLst = list()
    tempNoLst = list()
    for shp, siteNo in zip(shapeLst, siteNoLst):
        if siteNo not in maskLst:
            tempShpLst.append(shp)
            tempNoLst.append(siteNo)
    shapeLst = tempShpLst
    siteNoLst = tempNoLst

if iEnd == 0:  # do mask for every basin
    iEnd = len(siteNoLst)
    iStart = 0


t0 = time.time()
for k in range(iStart, iEnd):
    t1 = time.time()
    k = siteNoLst.index(siteNo)
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog)
    outFile = os.path.join(saveDir, siteNoLst[k])
    np.savez_compressed(outFile, mask=mask)
    t2 = time.time()
    print('basin {}/{} {:.2f} {:.2f}'.format(k-iStart, iEnd-iStart, t2-t1, t2-t0))
