import numpy as np
from hydroDL import utils
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

# sn = 0.0001
sn = 1e-5

dictTran = dict(mtdLst=None, statLn=None, statQT=None)


def transIn(dataIn, *, mtdLst=list(), statLn=None, statQT=None):
    data = dataIn.copy()
    # log
    indLog, mtdLog = (list(), list())
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[0] == 'log' or temp[0] == 'log2':
            indLog.append(k)
            mtdLog.append(temp[0])
    data[..., indLog] = logIn(data[..., indLog], mtdLog)
    # Linear
    indLn, mtdLn = (list(), list())
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'norm' or temp[-1] == 'stan':
            indLn.append(k)
            mtdLn.append(temp[-1])
    if len(indLn) > 0:
        data[..., indLn], statLn = linearIn(
            data[..., indLn], mtdLn, statIn=statLn)
    # QT
    indQT = list()
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'QT':
            indQT.append(k)
    if len(indQT) > 0:
        data[..., indQT], statQT = qtIn(data[..., indQT], statIn=statQT)
    dictOut = dict(mtdLst=mtdLst, statLn=statLn, statQT=statQT)
    return data, dictOut


def transOut(dataIn, dictTran):
    data = dataIn.copy()
    mtdLst = dictTran['mtdLst']
    statLn = dictTran['statLn']
    statQT = dictTran['statQT']
    # Linear
    indLn = list()
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'norm' or temp[-1] == 'stan':
            indLn.append(k)
    if len(indLn) > 0:
        data[..., indLn] = linearOut(data[..., indLn], statLn)
    # QT
    indQT = list()
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'QT':
            indQT.append(k)
    if len(indQT) > 0:
        data[..., indQT] = qtOut(data[..., indQT], statQT)
    # log
    indLog, mtdLog = (list(), list())
    for k, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[0] == 'log' or temp[0] == 'log2':
            indLog.append(k)
            mtdLog.append(temp[0])
    data[..., indLog] = logOut(data[..., indLog], mtdLog)
    return data


def linearIn(dataIn, mtdLst, statIn=None):
    data = dataIn.copy()
    noStat = True if statIn is None else False
    # calculate vS and out = (in-vS[0])/vS[1]
    if noStat:
        vS = np.ndarray([data.shape[-1], 2])
        for i, mtd in enumerate(mtdLst):
            if mtd == 'norm':
                v1 = np.nanpercentile(data[..., i], 15)
                v2 = np.nanpercentile(data[..., i], 85)
                stat = [(v2+v1)/2, (v2-v1)/2]
            if mtd == 'stan':
                stat = [np.nanmean(data[..., i]), np.nanstd(data[..., i])]
            vS[i, :] = stat
    else:
        vS = statIn.copy()

    for i in range(data.shape[-1]):
        # turn out to be faster than (data-vS0)/vS1
        vS[i, 1] = 1 if vS[i, 1] == 0 else vS[i, 1]
        data[..., i] = (data[..., i]-vS[i, 0])/vS[i, 1]
    return data, vS


def linearOut(dataIn, statIn):
    data = dataIn.copy()
    vS = statIn.copy()
    out = np.full(data.shape, np.nan)
    # turn out to be faster than (data-vS0)/vS1
    for i in range(data.shape[-1]):
        out[..., i] = data[..., i]*vS[i, 1]+vS[i, 0]
    return out


def qtIn(dataIn, statIn=None, nq=50):
    temp = dataIn.copy().reshape(-1, dataIn.shape[-1])
    if statIn is None:
        qt = QuantileTransformer(
            n_quantiles=nq, random_state=0, output_distribution='uniform')
        qt.fit(temp)
    else:
        qt = statIn
    outTemp = qt.transform(temp)
    outData = outTemp.reshape(dataIn.shape)
    outStat = qt
    return outData, outStat


def qtOut(dataIn, statIn):
    qt = statIn
    temp = dataIn.copy().reshape(-1, dataIn.shape[-1])
    outTemp = qt.inverse_transform(temp)
    out = outTemp.reshape(dataIn.shape)
    return out


def logIn(dataIn, mtdLst):
    data = dataIn.copy()
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd == 'log']
    data[..., indLog] = np.log(data[..., indLog]+sn)
    indLog2 = [i for i, mtd in enumerate(mtdLst) if mtd == 'log2']
    temp = data[..., indLog2].copy()
    temp[temp > 0] = np.log(temp[temp > 0]+1)
    temp[temp < 0] = -np.log(-temp[temp < 0]+1)
    data[..., indLog2] = temp
    return data


def logOut(dataIn, mtdLst):
    data = dataIn.copy()
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd == 'log']
    data[..., indLog] = np.exp(data[..., indLog])-sn
    indLog2 = [i for i, mtd in enumerate(mtdLst) if mtd == 'log2']
    temp = data[..., indLog2].copy()
    temp[temp > 0] = np.exp(temp[temp > 0])-1
    temp[temp < 0] = -np.exp(-temp[temp < 0])+1
    data[..., indLog2] = temp
    return data
