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
