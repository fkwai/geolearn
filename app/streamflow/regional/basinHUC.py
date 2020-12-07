import shapefile
from shapely.geometry import Point, shape
import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import gis, grid
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import time
import csv
import os
import pandas as pd
import numpy as np


# read database crd
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
fieldLst = ['HUC02', 'HUC04']
dfCode = pd.DataFrame(index=siteNoLstAll, columns=fieldLst)
dfCode.index.name = 'siteNo'

# read shapefile
fileHUC2 = r'C:\Users\geofk\work\map\HUC\HUC2_CONUS.shp'
fileHUC4 = r'C:\Users\geofk\work\map\HUC\HUC4_CONUS.shp'
fileShpLst = [fileHUC2, fileHUC4]
fieldShpLst = ['huc2', 'huc4']
for fileShp, fieldShp, fieldDf in zip(fileShpLst, fieldShpLst, fieldLst):
    shapeLst = shapefile.Reader(fileShp).shapes()
    shapeRecLst = shapefile.Reader(fileShp).records()
    # calculate inside polygon
    indLst = gis.pointInPoly(lat, lon, shapeLst)
    attrShp = [x[fieldShp] for x in shapeRecLst]
    attr = [attrShp[x]for x in indLst]
    dfCode[fieldDf] = attr

dfCode.to_csv(os.path.join(dirInv, 'ecoregion', 'basinHUC'))
