import numpy as np


def transIn(data, mtd, stat=None):
    noStat = True if stat is None else False
    if mtd == 'log-norm':
        temp = np.log(data+1)
        if stat is None:
            stat = [np.nanpercentile(temp, 10),
                    np.nanpercentile(temp, 90)]
        out = (temp-stat[0])/(stat[1]-stat[0])
    elif mtd == 'log-stan':
        temp = np.log(data+1)
        if stat is None:
            stat = [np.nanmean(temp), np.nanstd(temp)]
        out = (temp-stat[0])/stat[1]
    elif mtd == 'norm':
        if stat is None:
            stat = [np.nanpercentile(data, 10),
                    np.nanpercentile(data, 90)]
        out = (data-stat[0])/(stat[1]-stat[0])
    elif mtd == 'stan':
        if stat is None:
            stat = [np.nanmean(data), np.nanstd(data)]
        out = (data-stat[0])/stat[1]
    if noStat:
        return out, stat
    else:
        return out


def transOut(data, mtd, stat):
    if mtd == 'log-norm':
        temp = data*(stat[1]-stat[0])+stat[0]
        out = np.exp(temp)-1
    elif mtd == 'log-stan':
        temp = data*stat[1]+stat[0]
        out = np.exp(temp)-1
    elif mtd == 'norm':
        out = data*(stat[1]-stat[0])+stat[0]
    elif mtd == 'stan':
        out = data*stat[1]+stat[0]
    return out


def transInAll(data, mtdLst, statLst=None):
    noStat = True if statLst is None else False
    # find colums that need to do log
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd.split('-')[0] == 'log']
    data[..., indLog] = np.log(data[..., indLog]+1)
    # calculate stat
    vS = np.ndarray([data.shape[-1], 2])
    if noStat:
        statLst = list()
        for i, mtd in enumerate(mtdLst):
            temp = mtd.split('-')
            if temp[-1] == 'norm':
                stat = [np.nanpercentile(data[..., i], 10),
                        np.nanpercentile(data[..., i], 90)]
            if temp[-1] == 'stan':
                stat = [np.nanmean(data[..., i]), np.nanstd(data[..., i])]
            statLst.append(stat)
    # calculate vS and out = (in-vS[0])/vS[1]
    for i, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'norm':
            vS[i, :] = [statLst[i][0], statLst[i][1]-statLst[i][0]]
        if temp[-1] == 'stan':
            vS[i, :] = [statLst[i][0], statLst[i][1]]
    for i in range(data.shape[-1]):
        # turn out to be faster than (data-vS0)/vS1
        data[..., i] = (data[..., i]-vS[i, 0])/vS[i, 1]
    if noStat:
        return data, statLst
    else:
        return data


def transOutAll(data, mtdLst, statLst=list()):
    vS = np.ndarray([data.shape[-1], 2])
    # calculate vS and out = (in-vS[0])/vS[1]
    for i, mtd in enumerate(mtdLst):
        temp = mtd.split('-')
        if temp[-1] == 'norm':
            vS[i, :] = [statLst[i][0], statLst[i][1]-statLst[i][0]]
        if temp[-1] == 'stan':
            vS[i, :] = [statLst[i][0], statLst[i][1]]
    # turn out to be faster than (data-vS0)/vS1
    for i in range(data.shape[-1]):
        data[..., i] = data[..., i]*vS[i, 1]+vS[i, 0]
    # find colums that need to do log
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd.split('-')[0] == 'log']
    data[..., indLog] = np.exp(data[..., indLog])-1

    return data
