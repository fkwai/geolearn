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

# read shapefile
fileEcoShp = r'C:\Users\geofk\work\map\ecoRegion\NA_CEC_Eco_Level3\NA_CEC_Eco_Level3_Project.shp'

shapeLst = shapefile.Reader(fileEcoShp).shapes()
shapeRecLst = shapefile.Reader(fileEcoShp).records()
codeLstTemp = [x['NA_L3CODE'].split('.') for x in shapeRecLst]
codeLst = [[x[0] for x in codeLstTemp], [x[1] for x in codeLstTemp],
           [x[2] for x in codeLstTemp]]


# read database crd
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values

# calculate inside poligon
indLst = gis.pointInPoly(lat, lon, shapeLst)
codeSiteMat = np.ndarray([len(siteNoLstAll), 3])
for k in range(3):
    codeSite = [int(codeLst[k][x]) for x in indLst]
    codeSiteMat[:, k] = codeSite
dfCode = pd.DataFrame(index=siteNoLstAll, data=codeSiteMat,
                      columns=['code0', 'code1', 'code2'])
dfCode.index.name = 'siteNo'
dfCode.to_csv(os.path.join(dirInv,'ecoregion','basinCode'))
