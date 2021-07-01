import importlib
import fiona
import os
import time
import argparse
from osgeo import gdal, ogr, osr
import shapefile
from shapely.geometry import Point, Polygon, shape, box
from hydroDL.utils import gis
from hydroDL import kPath
import numpy as np
import pandas as pd


basinShp = os.path.join(kPath.dirData, 'USGS',
                        'basins', 'basinAll.shp')
rasterTiff = os.path.join(kPath.dirData, 'GLiM', 'NA_gageII_1KM.tif')

# get shapefile crs
os.environ['GDAL_DATA'] = r'C:\Users\geofk\Anaconda3\envs\pyTorch\Library\share\gdal'
shpFile = ogr.Open(basinShp)
layer = shpFile.GetLayer()
prjShape = layer.GetSpatialRef()
ds = gdal.Open(rasterTiff)
prjRaster = osr.SpatialReference()
prjRaster.ImportFromWkt(ds.GetProjection())
transform = osr.CoordinateTransformation(prjRaster, prjShape)
ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
nx = ds.RasterXSize
ny = ds.RasterYSize
# if xskew and yskew are 0
x = ulx+xres*np.arange(nx)+xres/2
y = uly+yres*np.arange(ny)+yres/2


# forturnately it is a regular raster so only need x and y. see testCrd.py
xP = np.full([nx], np.nan)
yP = np.full([ny], np.nan)
for i in range(nx):
    (xp, yp, z) = transform.TransformPoint(x[i], 0)
    xP[i] = xp
for j in range(ny):
    (xp, yp, z) = transform.TransformPoint(0, y[j])
    yP[j] = yp

raster = ds.GetRasterBand(1).ReadAsArray()
sf = shapefile.Reader(basinShp)
shapeLst = sf.shapes()
recLst = sf.records()

siteNoLst = [rec[2] for rec in recLst]
saveDir = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'mask')
t0 = time.time()
for k in range(len(shapeLst)):
    t1 = time.time()
    geog = shape(shapeLst[k])
    mask = gis.gridMask(yP, xP, geog)
    print('basin {} {:.2f} {:.2f}'.format(k, time.time()-t1, time.time()-t0))
    outFile = os.path.join(saveDir, siteNoLst[k])
    np.savez_compressed(outFile, mask)
print('total time {}'.format(time.time() - t0))
