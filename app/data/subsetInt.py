from hydroDL import pathSMAP
from hydroDL.utils import grid, flatData
from hydroDL.data import dbCsv
from hydroDL.post import plot
import numpy as np
import matplotlib.pyplot as plt

rootDB = pathSMAP['DB_L3_NA']
tRange = [20150401, 20180331]
df = dbCsv.DataframeCsv(rootDB=rootDB, subset='CONUS', tRange=tRange)
lat, lon = df.getGeo()
indGrid, uy, ux = grid.array2grid(df.indSub, lat=lat, lon=lon,fillMiss=False)
indSub = flatData(indGrid[::3, ::3]).astype(int)
df.subsetInit('CONUSv3f1', ind=indSub)

fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(lon, lat, 'bo')
plt.plot(lon[indSub], lat[indSub], 'r*')
fig.show()


# can not repeat what MATLAB did
indSub = flatData(indGrid[::4, 1::4]).astype(int)
dfSub = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=tRange)
latSub,lonSub=dfSub.getGeo()
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(lon[indSub], lat[indSub], 'r*')
plt.plot(lonSub, latSub, 'bo')
fig.show()
