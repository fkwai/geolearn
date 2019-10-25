import shapefile
from shapely.geometry import Point, shape
import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import gis, grid
from hydroDL.post import plot
import time
import csv

# read shapefile
# fileEcoShp = r'C:\Users\geofk\work\map\ecoRegion\NA_CEC_Eco_Level3\NA_CEC_Eco_Level3_Project.shp'
fileEcoShp = '/mnt/sdb/Kuai/map/ecoRegion/NA_CEC_Eco_Level3/NA_CEC_Eco_Level3_Project.shp'
shapeLst = shapefile.Reader(fileEcoShp).shapes()
shapeRecLst = shapefile.Reader(fileEcoShp).records()
codeLstTemp = [x['NA_L3CODE'].split('.') for x in shapeRecLst]
codeLst = [[x[0] for x in codeLstTemp], [x[1] for x in codeLstTemp],
           [x[2] for x in codeLstTemp]]

# read database crd
rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS', tRange=tRange)
lat, lon = df.getGeo()

# calculate inside poligon
# indLst=gis.pointInPoly(lat,lon,shapeLst)
# df.saveDataConst(indLst, 'ecoShapeInd', bWriteStat=True)
indLst = df.getDataConst('ecoShapeInd').squeeze().astype(int).tolist()
shapeCodeLst = [shapeRecLst[x]['NA_L3CODE'].split('.') for x in indLst]
for k in range(3):
    data = [int(x[k]) for x in shapeCodeLst]
    # df.saveDataConst(data, 'ecoRegionL'+str(k+1), bCalStat=False)

# subset
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False)
dataGrid, uy, ux = grid.array2grid(codeLst[0], lat=lat, lon=lon)
fig, ax = plot.plotMap(dataGrid, lat=uy, lon=ux)
