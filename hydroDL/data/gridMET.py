import netCDF4
import numpy as np


def readNcInfo(file):
    fh = netCDF4.Dataset(file)
    lon = fh.variables['lon'][:].data
    lat = fh.variables['lat'][:].data
    day = fh.variables['day'][:].data
    t = np.datetime64('1900-01-01') + np.array(day, dtype='timedelta64[D]')
    return t, lat, lon

def readNcData(file):
    fh = netCDF4.Dataset(file)
    var = list(fh.variables.keys())[-1]
    data = fh.variables[var][:].data
    mask = fh.variables[var][:].mask
    data[mask] = np.nan
    return data

