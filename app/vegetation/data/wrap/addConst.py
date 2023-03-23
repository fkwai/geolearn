import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json
import GDAL
from osgeo import ogr, gdal


outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
# load data
data = np.load(outFile)
x = data['x']
y = data['y']
t = data['t']
varX = data['varX']
varY = data['varY']


def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int)  # x pixel
    py = ((my - gt[3]) / gt[5]).astype(int)  # y pixel
    return data[py, px]


dictConst = {
    'slope': 'usa_slope_project.tif',
    'dem': 'usa_dem.tif',
    'canopyHeight': 'canopy_height.tif',
}
siteFile = os.path.join(kPath.dirVeg, 'model', 'data', 'site.csv')
dfSite = pd.read_csv(siteFile)

lat = dfSite['lat'].values
lon = dfSite['lon'].values

xc = np.full([len(dfSite), len(dictConst)], np.nan)
for k, (key, value) in enumerate(dictConst.items()):
    temp = get_value(os.path.join(kPath.dirVeg, 'const', value), lon, lat)
    xc[:, k] = temp
varXC=np.array(list(dictConst.keys()))

outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
np.savez(outFile, varY=varY, y=y, varX=varX, x=x, t=t, varXC=varXC, xc=xc)
