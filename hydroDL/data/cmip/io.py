import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.utils.grid
import time
from datetime import datetime

folder = os.path.join(kPath.dirRaw, 'CMIP6')


def walkFile():
    folder = os.path.join(kPath.dirRaw, 'CMIP6')
    fileLst = os.listdir(folder)
    colLst = ['var', 'freq', 'model', 'exp', 'variant', 'grid', 'sd', 'ed']
    df = pd.DataFrame(columns=colLst)
    for file in fileLst:
        ss = file.split('_')
        if ss[-1][-3:] == '.nc':
            [sdS, edS] = ss[-1][:-3].split('-')
            vLst = ss[:-1]+[pd.to_datetime(sdS), pd.to_datetime(edS)]
            df.loc[len(df)] = vLst
    return df


def name2dict(filePath):
    fileName = filePath.split(os.sep)[-1]
    ss = fileName.split('_')
    if ss[-1][-3:] == '.nc':
        [sdS, edS] = ss[-1][:-3].split('-')
        vLst = ss[:-1]+[pd.to_datetime(sdS), pd.to_datetime(edS)]
        out = dict()
        colLst = ['var', 'freq', 'model', 'exp', 'variant', 'grid', 'sd', 'ed']
        for k, key in enumerate(colLst):
            out[key] = vLst[k]
    else:
        print('not a nc file')
    return out


def df2file(df):
    fileLst = list()
    fileNamePat = '{}_{}_{}_{}_{}_{}_{:%Y%m%d}-{:%Y%m%d}.nc'
    for index, row in df.iterrows():
        fileName = fileNamePat.format(*row.values)
        fileLst.append(os.path.join(folder, fileName))
    return fileLst


def findFile(dfFile=None, **dictS):
    if dfFile is None:
        dfFile = walkFile()
    temp = dfFile.copy()
    for key in dictS.keys():
        if key == 'sd':
            temp = temp[temp['ed'] >= dictS[key]]
        elif key == 'ed':
            temp = temp[temp['sd'] < dictS[key]]
        else:
            temp = temp[temp[key] == dictS[key]]
    return temp


def readCrd(ncFile, adjust=True):
    # WARN
    fh = netCDF4.Dataset(ncFile)
    lon = fh.variables['lon'][:].data
    lat = fh.variables['lat'][:].data
    if adjust is True:
        lat, lon = hydroDL.utils.grid.adjustGrid(None, lat=lat, lon=lon)
    return lat, lon


def readNcData(file):
    fh = netCDF4.Dataset(file)
    nameDict = name2dict(file)
    var = nameDict['var']
    data = fh.variables[var][:].data
    lon = fh.variables['lon'][:].data
    lat = fh.variables['lat'][:].data
    # # WARN smart code not working
    # tv = fh.variables['time'][:].data
    # tuStr = fh.variables['time'].units.split(' ')
    # [yy, mm, dd] = tuStr[2].split('-')  # WARN
    # sday = datetime(int(yy), int(mm), int(dd))
    # if tuStr[0] == 'days':
    #     t = np.datetime64(sday) + np.array(tv, dtype='timedelta64[D]')
    # elif tuStr[0] == 'hours':
    #     t = np.datetime64(sday) + np.array(tv, 'timedelta64[h]')
    if nameDict['freq'] == 'day':
        sd = nameDict['sd']
        ed = nameDict['ed']
        t = np.arange(sd, ed+np.timedelta64(1, 'D'),
                      np.timedelta64(1, 'D')).astype('datetime64[D]') 
    if data.shape == (len(t), len(lat), len(lon)):
        data = np.transpose(data, axes=(1, 2, 0))
    else:
        raise Exception('check data')
    if var == 'pr':
        data = data * 60*60*24
    if var in ['tas', 'tasmax', 'tasmin']:
        data = data - 273.15
    data, lat, lon = hydroDL.utils.grid.adjustGrid(data, lat=lat, lon=lon)
    return data, (lat, lon, t)


def readCMIP(dfFile=None, check=True, latR=None, lonR=None, **dictS):
    # example
    # readCMIP(var='pr', exp='hist-1950', model=modelName,
    # sd=np.datetime64('2010-01-01'), ed=np.datetime64('2015-01-01'))
    df = findFile(dfFile=None, **dictS)
    df = df.sort_values('sd')
    fileLst = df2file(df)
    if 'sd' in dictS.keys() and 'ed' in dictS.keys():
        tR = [dictS['sd'], dictS['ed']]
    else:
        tR = None
    if check is True:
        lat, lon = readCrd(fileLst[0])
        _, (lat0, lon0, _) = hydroDL.utils.grid.clipGrid(
            lat=lat, lon=lon, latR=latR, lonR=lonR)
    dataLst = list()
    tLst = list()
    for file in fileLst:
        t0 = time.time()
        data, (lat, lon, t) = readNcData(file)
        data, (lat, lon, t) = hydroDL.utils.grid.clipGrid(
            grid=data, lat=lat, lon=lon, t=t,
            latR=latR, lonR=lonR, tR=tR)
        if check is True:
            if not np.array_equal(lat0, lat):
                raise Exception('lat doesnot match')
            if not np.array_equal(lon0, lon):
                raise Exception('lat doesnot match')
            if not np.all(t[:-1] < t[1:]):
                raise Exception('t is not sorted')
        print('reading {} {:.3f}'.format(file, time.time()-t0))
        dataLst.append(data)
        tLst.append(t)
    data = np.concatenate(dataLst, axis=-1)
    t = np.concatenate(tLst, axis=-1)
    return data, lat0, lon0, t
