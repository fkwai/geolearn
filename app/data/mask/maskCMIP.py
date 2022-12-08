
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse
import hydroDL.data.cmip.io as io
from hydroDL.utils import gis
from hydroDL import kPath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=0)
    parser.add_argument('-E', dest='iEnd', type=int, default=9067)
    parser.add_argument('-R', dest='reMask', type=int, default=False)
    args = parser.parse_args()
    iStart = args.iStart
    iEnd = args.iEnd
    reMask = args.reMask

mLst = ['MPI-ESM1-2-XR', 'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']
d1 = np.datetime64('2010-01-01')
d2 = np.datetime64('2010-01-02')
modelName = 'EC-Earth3P-HR'
maskFolder = os.path.join(kPath.dirData, 'USGS', 'mask', 'CMIP', modelName)


if not os.path.exists(maskFolder):
    os.makedirs(maskFolder)
    df = io.walkFile()
    dfTemp = df[df['model'] == modelName]
    ncFile = io.df2file(dfTemp)[0]
    lat, lon = io.readCrd(ncFile)
    crdFile = os.path.join(maskFolder, 'crd.npz')
    np.savez(crdFile, lat=lat, lon=lon)
else:
    crdFile = os.path.join(maskFolder, 'crd.npz')
    temp = np.load(crdFile)
    lat = temp['lat']
    lon = temp['lon']


shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basin_CONUS_prj.shp')

sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]


# not a good idea
# if reMask is False:
#     maskLst = [f[:-4] for f in os.listdir(maskFolder) if f[-3:] == 'npz']
#     tempShpLst = list()
#     tempNoLst = list()
#     for shp, siteNo in zip(shapeLst, siteNoLst):
#         if siteNo not in maskLst:
#             tempShpLst.append(shp)
#             tempNoLst.append(siteNo)
#     shapeLst = tempShpLst
#     siteNoLst = tempNoLst

t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    k
    outFile = os.path.join(maskFolder, siteNo)
    if os.path.exists(outFile+'.npz'):
        continue
    t1 = time.time()
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog)
    t2 = time.time()
    print('basin {} {:.2f} {:.2f}'.format(k, t2-t1, t2-t0))
    np.savez_compressed(outFile, mask)
