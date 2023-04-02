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
import time

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = list(range(2016, 2022))
# read site
siteFile = os.path.join(kPath.dirVeg, 'model', 'data', 'site.csv')
dfSite = pd.read_csv(siteFile)
dfSite = dfSite.set_index('siteId').sort_index()

dfLst = list()
for var in varLst:
    dataLst = list()
    tLst = list()
    t0 = time.time()
    for yr in yrLst:
        ncFile = os.path.join(kPath.dirRaw, 'gridMET', '{}_{}.nc'.format(var, yr))
        data, (lat, lon, t) = hydroDL.data.gridMET.io.readNcData(ncFile)
        dataLst.append(data.astype('float32'))
        tLst.append(t)
        print('{} {} {:.3f}'.format(var, yr, time.time() - t0))
    dataAll = np.concatenate(dataLst, axis=-1, dtype=float)
    tAll = np.concatenate(tLst, axis=-1)
    # read forcing
    dfD = pd.DataFrame(index=tAll, columns=dfSite.index.values)
    for k, row in dfSite.iterrows():
        latS = row['lat']
        lonS = row['lon']
        iy = np.argmin(np.abs(lat - latS))
        ix = np.argmin(np.abs(lon - lonS))
        dfD[k] = dataAll[iy, ix, :]
    dfLst.append(dfD)

# save df to csv
for var, dfM in zip(varLst, dfLst):
    dfM.to_csv(os.path.join(kPath.dirVeg, 'forcings', '{}.csv'.format(var)))
