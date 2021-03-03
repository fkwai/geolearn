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

"""
test for 
1 convert a raster to points
2 project and write a shapefile
using osgeo shits
Failed. Easier way is to export points to a table. 
"""

basinShp = os.path.join(kPath.dirData, 'USGS',
                        'basins', 'basinAll_prj.shp')
rasterTiff = os.path.join(kPath.dirData, 'GLiM', 'NA_gageII_1KM.tif')
saveDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

# get shapefile crs
os.environ['GDAL_DATA'] = r'C:\Users\geofk\Anaconda3\envs\pyTorch\Library\share\gdal'
shpFile = ogr.Open(basinShp)
layer = shpFile.GetLayer()
prjShape = layer.GetSpatialRef()
ds = gdal.Open(rasterTiff)
band = ds.GetRasterBand(1)
prjRaster = osr.SpatialReference()
prjRaster.ImportFromWkt(ds.GetProjection())
transform = osr.CoordinateTransformation(prjRaster, prjShape)


ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
nx = ds.RasterXSize
ny = ds.RasterYSize
# if xskew and yskew are 0
x = ulx+xres*np.arange(nx)+xres/2
y = uly+yres*np.arange(ny)+yres/2

temp = list()
# for k in range(nx):
#     temp.append([x[k], y[0]])
#     temp.append([x[k], y[-1]])
# for k in range(ny):
#     temp.append([x[0], y[k]])
#     temp.append([x[-1], y[k]])
tempPrj = list()
tempDeg = list()
for j in range(0, ny, 20):
    for i in range(0, nx, 20):
        tempPrj.append([x[i], y[j]])
        (xp, yp, z) = transform.TransformPoint(x[i], y[j])
        tempDeg.append([xp, yp])

dfT1 = pd.DataFrame(columns=['x', 'y'], data=np.array(tempPrj))
dfT1.to_csv(os.path.join(kPath.dirData, 'GLiM', 'pTestPrj2.csv'))
dfT2 = pd.DataFrame(columns=['x', 'y'], data=np.array(tempDeg))
dfT2.to_csv(os.path.join(kPath.dirData, 'GLiM', 'pTestDeg2.csv'))


# create a point layer to check
osgeoSucks = True
if not osgeoSucks:
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    outShp = os.path.join(kPath.dirData, 'GLiM', 'pointTest.shp')
    if os.path.exists(outShp):
        shpDriver.DeleteDataSource(outShp)
    outDataSource = shpDriver.CreateDataSource(outShp)
    outLayer = outDataSource.CreateLayer(outShp, prjShape, ogr.wkbPoint)
    point = ogr.Geometry(ogr.wkbPoint)
    for k in range(nx):
        point.AddPoint(x[k], y[0])
        point.AddPoint(x[k], y[-1])
    for k in range(ny):
        point.AddPoint(x[0], y[k])
        point.AddPoint(x[-1], y[k])
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(point)
    outFeature.SetField('id', ogr.OFTString)
    outLayer.CreateFeature(outFeature)

transform = osr.CoordinateTransformation(prjRaster, prjShape)
transform.TransformPoint(1, 2)
