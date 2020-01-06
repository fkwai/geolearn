import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shapefile
import numpy as np
import pandas as pd
from hydroDL.utils import gis

fig, ax = plt.subplots(1, 1)

ecoShapeFile = r'C:\Users\geofk\work\map\ecoRegion\NA_CEC_Eco_Level3\NA_CEC_Eco_Level3_Project.shp'
shapeLst = shapefile.Reader(ecoShapeFile).shapes()
shapeRecLst = shapefile.Reader(ecoShapeFile).records()
codeLstTemp = [x['NA_L3CODE'].split('.') for x in shapeRecLst]
codeLst = [[x[0] for x in codeLstTemp], [x[1] for x in codeLstTemp],
           [x[2] for x in codeLstTemp]]

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

subsetFile = r'D:\rnnSMAP\Database_SMAPgrid\Daily_L3_NA\Subset\ecoRegion{:02d}.csv'.format(
    ecoId)
crdC = pd.read_csv(
    r'D:\rnnSMAP\Database_SMAPgrid\Daily_L3_NA\CONUS\crd.csv').values
subsetInd = pd.read_csv(subsetFile).values.flatten()
xx = crdC[subsetInd, 1]
yy = crdC[subsetInd, 0]
mm.plot(xx, yy, '*b')

indLst=gis.pointInPoly(yy,xx,[shape])
temp=[ind==0 for ind in indLst]
mm.plot(xx[temp], yy[temp], '*g')
