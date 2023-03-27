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
    'sand': 'Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif',
    'clay': 'Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif',
    'silt': 'Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif',
}
siteFile = os.path.join(kPath.dirVeg, 'model', 'data', 'site.csv')
dfSite = pd.read_csv(siteFile)

lat = dfSite['lat'].values
lon = dfSite['lon'].values
xc = np.full([len(dfSite), len(dictConst)], np.nan)
for k, (key, value) in enumerate(dictConst.items()):
    temp = get_value(os.path.join(kPath.dirVeg, 'const', value), lon, lat)
    xc[:, k] = temp
varXC = np.array(list(dictConst.keys()))

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

matXC=np.concatenate([xc, matLC], axis=-1)
varXC=np.concatenate([varXC, np.array(['lc'+str(x) for x in range(1,10)])])

outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
np.savez(outFile, varY=varY, y=y, varX=varX, x=x, t=t, varXC=varXC, xc=matXC)
