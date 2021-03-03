import fiona
import os
import time
import argparse
from osgeo import gdal, ogr, osr
import shapefile
from hydroDL.utils import gis
from hydroDL import kPath
import numpy as np
import pandas as pd
from shapely.geometry import shape



basinShp = os.path.join(kPath.dirData, 'USGS',
                        'basins', 'basinN5.shp')
rasterTiff = os.path.join(kPath.dirData, 'GLiM', 'NA_gageII_1KM.tif')
saveDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

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
raster = ds.GetRasterBand(1).ReadAsArray()

#  a test. Looks good
temp = list()
for j in range(1000, 1100):
    for i in range(2000, 2100):
        (xp, yp, z) = transform.TransformPoint(x[i], y[j])
        temp.append([xp, yp, raster[j, i]])
dfT2 = pd.DataFrame(columns=['x', 'y', 'data'], data=np.array(temp))
dfT2.to_csv(os.path.join(kPath.dirData, 'GLiM', 'pTest_basinN5.csv'))

# create mask
xM = np.full([ny, nx], np.nan)
yM = np.full([ny, nx], np.nan)
for j in range(ny):
    print(j)
    for i in range(nx):
        (xp, yp, z) = transform.TransformPoint(x[i], y[j])
        xM[j, i] = xp
        yM[j, i] = yp
# forturnately it is a regular raster
np.unique(xM[:, np.random.randint(nx)])
np.unique(yM[np.random.randint(ny), :])
# so only need x and y

xP = np.full([nx], np.nan)
yP = np.full([ny], np.nan)
for i in range(nx):
    (xp, yp, z) = transform.TransformPoint(x[i], 0)
    xP[i] = xp
for j in range(ny):
    (xp, yp, z) = transform.TransformPoint(0, y[j])
    yP[j] = yp

np.unique(xP-xM[np.random.randint(ny), :])
np.unique(yP-yM[:, np.random.randint(nx)])

# test for generate a mask
raster = ds.GetRasterBand(1).ReadAsArray()
sf = shapefile.Reader(basinShp)
shapeLst = sf.shapes()
recLst = sf.records()
#  convert the mask to a geotiff

k = 8
geog = shape(shapeLst[k])
t0 = time.time()
mask = gis.gridMask(yP, xP, geog, calArea=False)
print(time.time()-t0)
t0 = time.time()
mask = gis.gridMask(yP, xP, geog, calArea=True)
print(time.time()-t0)
saveDir = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'mask')

outRasterFile = rasterTiff = os.path.join(
    kPath.dirData, 'GLiM', 'testMask.tif')
driver = gdal.GetDriverByName('GTiff')
outRaster = driver.Create(outRasterFile, nx, ny, 1, gdal.GDT_Byte)
outRaster.SetGeoTransform((ulx, xres, xskew, uly, yskew, yres))
outband = outRaster.GetRasterBand(1)
outband.WriteArray(mask)
outRaster.SetProjection(prjRaster.ExportToWkt())
outband.FlushCache()
del outRaster