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
    norm = np.full(data.shape, np.nan)
    if statLst is None:
        statLst = list()
        for k, mtd in enumerate(mtdLst):
            if len(data.shape) == 3:
                norm[:, :, k], stat = transIn(data[:, :, k], mtd)
            elif len(data.shape) == 2:
                norm[:, k], stat = transIn(data[:, :, k], mtd)
            statLst.append(stat)
        return norm, statLst
    else:
        for k, mtd in enumerate(mtdLst):
            if len(data.shape) == 3:
                norm[:, :, k] = transIn(data[:, :, k], mtd, stat=statLst[k])
            elif len(data.shape) == 2:
                norm[:, k] = transIn(data[:, :, k], mtd, stat=statLst[k])
        return norm, statLst
