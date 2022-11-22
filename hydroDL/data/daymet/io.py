
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.utils.grid
import time
from datetime import datetime

dirVal = os.path.join(
    kPath.dirRaw, 'daymet', 'Daymet_xval_V4R1_2132', 'data')


def fileVal(var, yr):
    ncFile = os.path.join(
        dirVal, 'daymet_v4_stnxval_{}_na_{}.nc'.format(var, yr))
    return ncFile


def readInfo(ncFile):
    # temporal
    fh = netCDF4.Dataset(ncFile)
    id = fh.variables['station_id'][:].data
    idLst = [bytes.decode(s) for s in np.frombuffer(id[0], dtype='|S15')]
    return idLst


def readNcVal(ncFile):
    fh = netCDF4.Dataset(ncFile)
    lon = fh.variables['stn_lon'][:].data
    lat = fh.variables['stn_lat'][:].data
    data = fh.variables['obs'][:].data
    data[data == -9999] = np.nan
    tDay = fh.variables['time'][:].data
    t = np.datetime64('1950-01-01') + np.array(tDay, dtype='timedelta64[D]')
    return data, (lat, lon, t)


def readVal(var, syr, eyr, dtype=None):
    yrLst = list(range(syr, eyr))
    dataLst = list()
    tLst = list()
    for yr in yrLst:
        t0 = time.time()
        ncFile = fileVal(var, yr)
        data, (lat, lon, t) = readNcVal(ncFile)
        if dtype is not None:
            data = data.astype(dtype)
        print('reading {} {:.3f}'.format(ncFile, time.time()-t0))
        dataLst.append(data)
        tLst.append(t)
    data = np.concatenate(dataLst, axis=-1, dtype=float)
    t = np.concatenate(tLst, axis=-1)
    return data, (lat, lon, t)
