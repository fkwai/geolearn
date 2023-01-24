import numpy as np
import netCDF4
from hydroDL.utils import grid
from hydroDL import kPath
import os
import time


def readNcInfo(file):
    fh = netCDF4.Dataset(file)
    lon = fh.variables['lon'][:].data
    lat = fh.variables['lat'][:].data
    day = fh.variables['day'][:].data
    t = np.datetime64('1900-01-01') + np.array(day, dtype='timedelta64[D]')
    return t, np.asarray(lat), np.asarray(lon)


def readNcData(file):
    fh = netCDF4.Dataset(file)
    var = list(fh.variables.keys())[-1]
    data = fh.variables[var][:].data
    mask = fh.variables[var][:].mask
    lon = fh.variables['lon'][:].data
    lat = fh.variables['lat'][:].data
    data[mask] = np.nan
    day = fh.variables['day'][:].data
    t = np.datetime64('1900-01-01') + np.array(day, dtype='timedelta64[D]')
    if data.shape == (len(t), len(lat), len(lon)):
        data = np.transpose(data, axes=(1, 2, 0))
    else:
        raise Exception('check order')
    if var in ['tmmx', 'tmmn']:
        data = data - 273.15
    data, lat, lon = grid.adjustGrid(data, lat=lat, lon=lon)
    return data, (lat, lon, t)


def read(var, syr, eyr, dtype=None):
    yrLst = list(range(syr, eyr))
    dataLst = list()
    tLst = list()
    for yr in yrLst:
        t0 = time.time()
        ncFile = os.path.join(kPath.dirRaw, 'gridMET',
                              '{}_{}.nc'.format(var, yr))
        data, (lat, lon, t) = readNcData(ncFile)
        if dtype is not None:
            data = data.astype(dtype)
        print('reading {} {:.3f}'.format(ncFile, time.time()-t0))

        dataLst.append(data)
        tLst.append(t)
    data = np.concatenate(dataLst, axis=-1, dtype=float)
    t = np.concatenate(tLst, axis=-1)
    return data, (lat, lon, t)
