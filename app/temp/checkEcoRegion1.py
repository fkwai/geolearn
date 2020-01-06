

import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shapefile
import numpy as np
import pandas as pd
from hydroDL.utils import gis
import hydroDL
from hydroDL.data import dbCsv

fig, ax = plt.subplots(1, 1)

ecoShapeFile = r'C:\Users\geofk\work\map\ecoRegion\comb\ecoRegion.shp'
shapeLst = shapefile.Reader(ecoShapeFile).shapes()
shapeRecLst = shapefile.Reader(ecoShapeFile).records()
ecoIdLst = [rec[1] for rec in shapeRecLst]


ecoId = 6
shape = shapeLst[ecoIdLst.index(ecoId)]
mm = Basemap(llcrnrlat=25, urcrnrlat=50,
             llcrnrlon=-125, urcrnrlon=-65,
             projection='cyl', resolution='c', ax=ax)
mm.drawcoastlines()
crd = np.array(shape.points)
par = list(shape.parts)
par.append(len(crd))
if len(par) > 1:
    for k in range(0, len(par)-1):
        # for k in range(0, 2):
        x = crd[par[k]:par[k+1], 0]
        y = crd[par[k]:par[k+1], 1]
        mm.plot(x, y, color='r', linewidth=2)

subsetFile = r'D:\rnnSMAP\Database_SMAPgrid\Daily_L3_NA\Subset\ecoRegion{:02d}_v2f1.csv'.format(
    ecoId)
# read database crd
rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='ecoRegion{:02d}_v2f1'.format(ecoId), tRange=tRange)
lat, lon = df.getGeo()

mm.plot(lon, lat, '*b')
fig.show()