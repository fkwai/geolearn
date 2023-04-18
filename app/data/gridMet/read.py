import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

# read gridMet
dataLst = list()
tLst = list()
yrLst = list(range(2010, 2020))
sd = np.datetime64('2010-01-01')
ed = np.datetime64('2020-01-01')

for yr in yrLst:
    ncFile = os.path.join(kPath.dirRaw, 'gridMET',
                          '{}_{}.nc'.format('pr', yr))
    data, (lat, lon, t) = hydroDL.data.gridMET.io.readNcData(ncFile)
    dataLst.append(data.astype('float32'))
    tLst.append(t)
dataAll = np.concatenate(dataLst, axis=-1, dtype=float)
tAll = np.concatenate(tLst, axis=-1)


fig = plt.figure()
gs = gridspec.GridSpec(1, 1)
ax = mapplot.mapGrid(fig, gs[0, 0], lat, lon,
                     np.mean(data, axis=-1)*365)
fig.show()

fh = netCDF4.Dataset(ncFile)
fh.variables