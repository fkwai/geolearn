import shapefile
from hydroDL.utils import gis
import time
import pandas as pd
import numpy as np
import hydroDL
from hydroDL.data import dbCsv
from mpl_toolkits.basemap import Basemap

# read shapefile
ecoShapeFile = r'C:\Users\geofk\work\map\ecoRegion\comb\ecoRegion.shp'
shapeLst = shapefile.Reader(ecoShapeFile).shapes()
shapeRecLst = shapefile.Reader(ecoShapeFile).records()
ecoIdLst = [rec[1] for rec in shapeRecLst]


# read database crd
rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
lat, lon = df.getGeo()


indLst = gis.pointInPoly(lat, lon, shapeLst)

for k in range(17):
    ind = [i for i, x in enumerate(indLst) if x == k]
    ecoId = ecoIdLst[k]
    df.subsetInit('ecoRegion{}_v2f1'.format(ecoId), ind=ind)

fig, ax = plt.subplots(1, 1)
kk = 11
shape = shapeLst[kk]
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

temp = [ind == kk for ind in indLst]
xx = lon[temp]
yy = lat[temp]
mm.plot(xx, yy, '*b')
fig.show()
