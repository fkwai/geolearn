from hydroDL.post import mapplot, axplot, figplot
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.utils.grid
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dirVal = os.path.join(
    kPath.dirRaw, 'daymet', 'Daymet_xval_V4R1_2132', 'data')

yr = 2010
var = 'prcp'
ncFile = os.path.join(dirVal, 'daymet_v4_stnxval_{}_na_{}.nc'.format(var, yr))
fh = netCDF4.Dataset(ncFile)

fh.variables

fh.variables.keys()
lon = fh.variables['stn_lon'][:].data
lat = fh.variables['stn_lat'][:].data
data = fh.variables['obs'][:].data
data[data == -9999] = np.nan
tDay = fh.variables['time'][:].data
t = np.datetime64('1950-01-01') + np.array(tDay, dtype='timedelta64[D]')

# plot all site
fig = plt.figure(figsize=(8, 4))
vR = [0.5, 1]
gs = gridspec.GridSpec(1, 1)
ax = mapplot.mapPoint(
    fig, gs[0, 0], lat, lon, np.nanmean(data[:, :10], axis=1), s=5)
fig.show()

id = fh.variables['station_id'][:].data

aa = [bytes.decode(s) for s in np.frombuffer(id[:], dtype='S15')]


[bytes.decode(s) for s in np.frombuffer(id[0], dtype='S1')]

idLst = list()
for v in id:
    idLst = idLst+[bytes.decode(s) for s in np.frombuffer(v, dtype='S255')]


aa = id.tobytes().decode('utf-8')
fh.variables['station_id']

aa = id.decode('utf-8')
