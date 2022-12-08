import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import hydroDL.utils.ts
import hydroDL.utils.stat
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse
import hydroDL.data.gridMET.io as io
from hydroDL.utils import gis
from hydroDL import kPath
import pandas as pd


shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'gage_CONUS_prj.shp')
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
data = [rec[:7] for rec in recLst]
colLst = ['siteNo', 'name', 'class','ecoRegion', 'area', 'HUC02', 'lat', 'lon']
df = pd.DataFrame(data, columns=colLst)
df = df.set_index('siteNo')
fileSite = os.path.join(kPath.dirData, 'USGS', 'basins', 'siteCONUS.csv')
df.to_csv(fileSite)


from hydroDL.data import gageII
fileSite = os.path.join(kPath.dirData, 'USGS', 'basins', 'siteCONUS.csv')
modelName = 'MPI-ESM1-2-XR'

fileSite = os.path.join(kPath.dirData, 'USGS', 'basins', 'siteCONUS.csv')
dfSite = pd.read_csv(fileSite, dtype={'siteNo': str}).set_index('siteNo')

dirC = os.path.join(kPath.dirData, 'USGS', 'CMIP', modelName, 'output')
dirG = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'output')
siteLst1 = [f for f in os.listdir(dirC)]
siteLst2 = [f for f in os.listdir(dirG)]
siteLst = [v for v in siteLst1 if v in siteLst2]

dfS = dfSite.loc[siteLst]
lat = dfS['lat'].values
lon = dfS['lon'].values

data1 = np.zeros([14975, len(siteLst), 3])
data2 = np.zeros([10958, len(siteLst), 3])

for k, siteNo in enumerate(siteLst):
    print(k, siteNo)
    file1 = os.path.join(dirG, siteNo)
    df1 = pd.read_csv(file1)
    df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
    df1 = df1.set_index('date')
    v1 = df1[['pr', 'tmmn', 'tmmx']].values
    data1[:, k, :] = v1
    file2 = os.path.join(dirC, siteNo)
    df2 = pd.read_csv(file2)
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
    df2 = df2.set_index('date')
    v2 = df2[['pr', 'tasmin', 'tasmax']].values
    data2[:, k, :] = v2
t1 = df1.index.values
t2 = df2.index.values
[t, indT1, indT2] = np.intersect1d(t1, t2, return_indices=True)

d1 = np.swapaxes(data1[indT1, :, :], 2, 0)
d2 = np.swapaxes(data2[indT2, :, :], 2, 0)

k = 0

m1, _ = hydroDL.utils.ts.data2Monthly(d1[k, :, :], t, func='nansum')
m2, _ = hydroDL.utils.ts.data2Monthly(d2[k, :, :], t, func='nansum')
rD = hydroDL.utils.stat.gridCorrT(d1[k, :, :], d2[k, :, :])
rM = hydroDL.utils.stat.gridCorrT(m1, m2)


fig, ax = figplot.boxPlot([rD, rM], widths=0.5, cLst='rgbkm',
                          label1=['daily', 'monthly'], figsize=(4, 4))
fig.show()

area = dfS['HUC02'].values

fig, ax = plt.subplots(1, 1)
ax.plot(area, rD, 'r*')
ax.set_title('area [sqkm] vs daily correlation ')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(area, rM, 'b*')
ax.set_title('area [sqkm] vs monthly correlation ')
fig.show()
