import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from hydroDL import kPath
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
from hydroDL.post import mapplot, axplot, figplot
importlib.reload(hydroDL.data.cmip.io)
# get all files
df = hydroDL.data.cmip.io.walkFile()


modelName = 'MPI-ESM1-2-XR'
d1 = np.datetime64('2010-01-01')
d2 = np.datetime64('2015-01-01')
d3 = np.datetime64('2019-12-31')

data1, lat1, lon1, t1 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var='pr', exp='hist-1950',
    sd=d1, ed=d2, model=modelName)

data2, lat2, lon2, t2 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var='pr', exp='highres-future',
    sd=d2, ed=d3, model=modelName)

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapGrid(fig, gs[0, 0], lat1, lon1,
                     np.mean(data1, axis=-1), extent=None)
ax = mapplot.mapGrid(fig, gs[1, 0], lat2, lon2,
                     np.mean(data2, axis=-1), extent=None)
fig.show()

lat = lat1
lon = lon1

importlib.reload(figplot)
importlib.reload(mapplot)


def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapGrid(
        figM, gsM[0, 0], lat, lon, np.mean(data1, axis=-1),
        vRange=[0, 5])
    figP, axP = plt.subplots(1, 1)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    iY, iX = iP
    axplot.plotTS(axP, t1, data1[iY, iX, :])


figM, figP = figplot.clickMap(funcM, funcP)
