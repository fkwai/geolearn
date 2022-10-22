
import os
import numpy as np
import pandas as pd
from hydroDL import kPath
import json
__all__ = ['readStreamflow', 'readForcing', 'readAttr', 'siteNoLst']

# hard code time length
dirDB = os.path.join(kPath.dirData, 'camels')
dirG = os.path.join(dirDB, 'camels_attributes_v2.0')
dirQ = os.path.join(dirDB,
                    'basin_timeseries_v1p2_metForcing_obsFlow',
                    'basin_dataset_public_v1p2', 'usgs_streamflow')
fileVar = os.path.join(dirG, 'lookupVar.json')
fileVarCode = os.path.join(dirG, 'lookupCode.json')


def readInfo():
    gageFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'basin_metadata',
                            'gauge_information.txt')
    dictHead = {'huc02': str, 'usgsId': str, 'name': str,
                'lat': float, 'lon': float, 'area': float}
    dfInfo = pd.read_csv(gageFile, sep='\t', header=None, names=dictHead.keys(),
                         skiprows=1, dtype=dictHead)
    return dfInfo.set_index('usgsId')


def siteNoLst():
    dfInfo = readInfo()
    return dfInfo.index.tolist()


def readStreamflow(siteNo, *, dfInfo=None):
    if dfInfo is None:
        dfInfo = readInfo()
    huc = dfInfo.loc[siteNo]['huc02']
    area = dfInfo.loc[siteNo]['area']
    siteFile = os.path.join(dirQ, huc, '{}_streamflow_qc.txt'.format(siteNo))
    colLst = ['siteNo', 'Y', 'M', 'D', 'q', 'flagStr']
    df = pd.read_csv(siteFile, sep=r'\s+', header=None, names=colLst,
                     parse_dates=[['Y', 'M', 'D']])
    df.rename(columns={'Y_M_D': 'date'}, inplace=True)
    unitConv = 0.3048**3*24*60*60/1000
    df['runoff'] = df['q']/area*unitConv  # q[ft^3/s] area [sqkm] to mm/day
    df['flag'] = df['flagStr'].replace({'A': 0, 'A:e': 1, 'M': 2})
    out = df[['date', 'q', 'runoff', 'flag']].set_index('date')
    return out


def readForcing(siteNo, opt='nldas', *, dfInfo=None):
    if dfInfo is None:
        dfInfo = readInfo()
    huc = dfInfo.loc[siteNo]['huc02']
    dataFolder = os.path.join(
        dirDB,
        'basin_timeseries_v1p2_metForcing_obsFlow',
        'basin_dataset_public_v1p2',
        'basin_mean_forcing',
    )
    dictOpt = {'daymet': 'cida', 'nldas': 'nldas', 'maurer': 'maurer'}
    if opt not in dictOpt.keys():
        raise Exception('wrong dataset')
    dataFile = os.path.join(
        dataFolder, opt, huc,
        '{}_lump_{}_forcing_leap.txt'.format(siteNo, dictOpt[opt]),
    )
    try:
        df = pd.read_csv(dataFile, sep=r'\s+', header=0, skiprows=3,
                         parse_dates=[['Year', 'Mnth', 'Day']])
    except:
        print('sth wrong with {} {}'.format(siteNo, opt))
        colLst = ['Y', 'M', 'D', 'H', 'dayl', 'prcp',
                  'srad', 'swe', 'tmax', 'tmin', 'vp']
        df = pd.read_csv(dataFile, sep=r'\s+', header=None, skiprows=4,
                         names=colLst, parse_dates=[['Y', 'M', 'D']])
        df.rename(columns={'Y_M_D': 'date'}, inplace=True)
    df.columns = df.columns.str.lower()
    dictCol = {
        'year_mnth_day': 'date',
        'dayl(s)': 'dayl', 'prcp(mm/day)': 'prcp',
        'srad(w/m2)': 'srad',  'swe(mm)': 'swe',
        'tmax(c)': 'tmax', 'tmin(c)': 'tmin', 'vp(pa)': 'vp'}
    df.rename(columns=dictCol, inplace=True)
    colOut = ['date', 'dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    out = df[colOut].set_index('date')
    return out


def readAttr(*, siteNoLst=None, varLst=None):
    if os.path.exists(fileVar):
        with open(fileVar, 'r') as fp:
            dictVar = json.load(fp)
        with open(fileVarCode, 'r') as fp:
            dictCode = json.load(fp)
    else:
        initG()
    # read all as datasize is small
    tempLst = list()
    for grp in dictVar.keys():
        dataFile = os.path.join(dirG, 'camels_' + grp + '.txt')
        temp = pd.read_csv(dataFile, sep=';', dtype={'gauge_id': 'str'})
        temp.set_index('gauge_id', inplace=True)
        tempLst.append(temp)
    pdf = pd.concat(tempLst, axis=1)
    if siteNoLst is not None:
        pdf = pdf.loc[siteNoLst]
    if varLst is not None:
        pdf = pdf[varLst]
    pdf.replace(dictCode, inplace=True)
    return pdf


def initG():
    # initialize attribute folder - read all, code string values, write var list
    # only need to run once
    grpLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
    tempLst = list()
    dictVar = dict()
    for grp in grpLst:
        dataFile = os.path.join(dirG, 'camels_' + grp + '.txt')
        temp = pd.read_csv(dataFile, sep=';', dtype={'gauge_id': 'str'})
        temp.set_index('gauge_id', inplace=True)
        tempLst.append(temp)
        dictVar[grp] = temp.columns.tolist()
    pdf = pd.concat(tempLst, axis=1)
    fileVar = os.path.join(dirG, 'lookupVar.json')
    with open(fileVar, 'w') as fp:
        json.dump(dictVar, fp, indent=4)
    # update code
    dictCode = dict()
    varCodeLst = pdf.select_dtypes(include=['object']).columns.tolist()
    for var in varCodeLst:
        strLst = pdf[var].unique().tolist()
        if np.nan in strLst:
            strLst.remove(np.nan)
        strLst.sort()
        codeLst = list(range(len(strLst)))
        dictCode[var] = dict(zip(strLst, codeLst))
    with open(fileVarCode, 'w') as fp:
        json.dump(dictCode, fp, indent=4)
