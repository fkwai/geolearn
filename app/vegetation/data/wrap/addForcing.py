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
    dfD = pd.DataFrame(index=tAll, columns=dfSite['siteId'].values)
    for k, row in dfSite.iterrows():
        latS = row['lat']
        lonS = row['lon']
        iy = np.argmin(np.abs(lat - latS))
        ix = np.argmin(np.abs(lon - lonS))
        dfD[row['siteId']] = dataAll[iy, ix, :]
    dfM = dfD.resample('SM').mean()
    dfLst.append(dfM)

tM = dfM.index.values
sdN = '2016-08-31'
edN = '2021-12-15'
tN = pd.date_range(sdN, edN, freq='SM')
_, indT1, indT2 = np.intersect1d(tM, tN, return_indices=True)
matF = np.full([len(tN), len(dfSite), len(varLst)], np.nan)
for k, (var, dfM) in enumerate(zip(varLst, dfLst)):
    v = dfM.values[indT1, :]
    matF[:, :, k] = v

# save to forcing files


# append forcings
outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
# load data
data = np.load(outFile)
x = data['x']
y = data['y']
xc = data['xc']
t = data['t']
varX = data['varX']
varY = data['varY']
varXC = data['varXC']

x = np.concatenate([x, matF], axis=-1)
varX = list(varX) + varLst


np.savez(outFile, varY=varY, y=y, varX=varX, x=x, t=t, varXC=varXC, xc=xc)
