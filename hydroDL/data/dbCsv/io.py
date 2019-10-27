import os
import numpy as np
import pandas as pd
import time


def readDBtime(*, rootDB, rootName, yrLst):
    tnum = np.empty(0, dtype=np.datetime64)
    for yr in yrLst:
        timeFile = os.path.join(rootDB, rootName, str(yr), 'timeStr.csv')
        temp = (pd.read_csv(timeFile, dtype=str, header=None).astype(
            np.datetime64).values.flatten())
        tnum = np.concatenate([tnum, temp], axis=0)
    return tnum


def readVarLst(*, rootDB, varLst):
    varFile = os.path.join(rootDB, 'Variable', varLst + '.csv')
    varLst = pd.read_csv(
        varFile, header=None, dtype=str).values.flatten().tolist()
    return varLst


def readDataTS(*, rootDB, rootName, indSub, indSkip, yrLst, fieldName):
    tnum = readDBtime(rootDB=rootDB, rootName=rootName, yrLst=yrLst)
    nt = len(tnum)
    ngrid = len(indSub)

    # read data
    data = np.zeros([ngrid, nt])
    k1 = 0
    for yr in yrLst:
        t1 = time.time()
        dataFile = os.path.join(rootDB, rootName, str(yr), fieldName + '.csv')
        dataTemp = pd.read_csv(
            dataFile, dtype=np.float, skiprows=indSkip, header=None).values
        k2 = k1 + dataTemp.shape[1]
        data[:, k1:k2] = dataTemp
        k1 = k2
        print('read ' + dataFile, time.time() - t1)
    data[np.where(data == -9999)] = np.nan
    return data


def readDataConst(*, rootDB, rootName, indSub, indSkip, fieldName):
    # read data
    dataFile = os.path.join(rootDB, rootName, 'const', fieldName + '.csv')
    data = pd.read_csv(
        dataFile, dtype=np.float, skiprows=indSkip,
        header=None).values.flatten()
    data[np.where(data == -9999)] = np.nan
    return data


def readStat(*, rootDB, fieldName, isConst=False):
    if isConst is False:
        statFile = os.path.join(rootDB, 'Statistics', fieldName + '_stat.csv')
    else:
        statFile = os.path.join(rootDB, 'Statistics',
                                'const_' + fieldName + '_stat.csv')
    stat = pd.read_csv(statFile, dtype=np.float, header=None).values.flatten()
    return stat


def writeDataConst(data, fieldName, *, rootDB, subset, ndigit=8, bCalStat=True):
    pdf = pd.DataFrame(data)
    dataFile = os.path.join(rootDB, subset, 'const', fieldName + '.csv')
    pdf.to_csv(dataFile, header=False, index=False,
               float_format='%.{}f'.format(ndigit))
    print('writing data '+dataFile)

    statFile = os.path.join(rootDB, 'Statistics',
                            'const_' + fieldName + '_stat.csv')
    if not os.path.isfile(statFile):
        stat = calStat(pdf.values, bCalStat)
        pd.DataFrame(stat).to_csv(statFile, header=False, index=False,
                                  float_format='%.{}f'.format(ndigit))
        print('writing stat '+dataFile)


def calStat(data, bCalStat=True):
    stat = [0, 1, 0, 1]
    if bCalStat is True:
        stat[0] = np.percentile(data, 10)
        stat[1] = np.percentile(data, 90)
        stat[2] = np.nanmean(data)
        stat[3] = np.nanstd(data)
    return stat


def transNorm(data, stat, toNorm=True):
    if toNorm is True:
        dataOut = (data - stat[2]) / stat[3]
    else:
        dataOut = data * stat[3] + stat[2]
    return (dataOut)


def transNormSigma(data, stat, toNorm=True):
    if toNorm is True:
        dataOut = np.log((data / stat[3])**2)
    else:
        dataOut = np.sqrt(np.exp(data)) * stat[3]
    return (dataOut)
