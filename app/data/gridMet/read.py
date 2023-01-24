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
dataLst0 = list()
tLst0 = list()
yrLst = list(range(2010, 2020))
sd = np.datetime64('2010-01-01')
ed = np.datetime64('2020-01-01')

for yr in yrLst:
    ncFile = os.path.join(kPath.dirRaw, 'gridMET',
                          '{}_{}.nc'.format('pr', yr))
    data, lat, lon, t = hydroDL.data.gridMET.io.readNcData(ncFile)
    dataLst0.append(data.astype('float32'))
    tLst0.append(t)
data0 = np.concatenate(dataLst0, axis=0, dtype=float)
t0 = np.concatenate(tLst0, axis=-1)
_, lat0, lon0 = hydroDL.data.gridMET.io.readNcInfo(ncFile)
data0 = np.transpose(data0, axes=(1, 2, 0))

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)
ax = mapplot.mapGrid(fig, gs[0, 0], lat0, lon0,
                     np.mean(data0, axis=-1))
fig.show()
