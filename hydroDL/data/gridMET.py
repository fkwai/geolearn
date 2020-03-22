import netCDF4
import os
import numpy as np
import pandas as pd
from hydroDL import kPath

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']

dictStat = dict(pr='log-norm', sph='norm', srad='norm',
                tmmn='norm', tmmx='norm', pet='norm', etr='norm')


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
    data[mask] = np.nan
    return data


def readBasin(siteNo, varLst=varLst):
    """read basin averaged forcing data, plenty of work is done before. See:
    app\waterQual\data\gridMetExtract.py
    app\waterQual\data\gridMetFromRaw.py
    app\waterQual\data\gridMetMask.py
    Arguments:
        siteNo {str} -- usgs site number
    Returns:
        pandas.Dataframe -- output table
    """
    fileF = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'output', siteNo)
    dfF = pd.read_csv(fileF)
    dfF = dfF.set_index('date')
    return dfF[varLst]
