import numpy as np
from hydroDL import utils
# sn = 0.0001
sn = 1


def transIn(dataIn, mtdLst, statIn=None):
    data = dataIn.copy()
    noStat = True if statIn is None else False
    # find colums that need to do log
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd.split('-')[0] == 'log']
    data[..., indLog] = np.log(data[..., indLog]+sn)
    indLog2 = [i for i, mtd in enumerate(
        mtdLst) if mtd.split('-')[0] == 'log2']
    temp = data[..., indLog2].copy()
    temp[temp > 0] = np.log(temp[temp > 0]+1)
    temp[temp < 0] = -np.log(-temp[temp < 0]+1)
    data[..., indLog2] = temp

    # calculate vS and out = (in-vS[0])/vS[1]
    if noStat:
        vS = np.ndarray([data.shape[-1], 2])
        for i, mtd in enumerate(mtdLst):
            temp = mtd.split('-')
            if temp[-1] == 'norm':
                v1 = np.nanpercentile(data[..., i], 15)
                v2 = np.nanpercentile(data[..., i], 85)
                stat = [(v2+v1)/2, (v2-v1)/2]
            if temp[-1] == 'stan':
                stat = [np.nanmean(data[..., i]), np.nanstd(data[..., i])]
            vS[i, :] = stat
    else:
        vS = statIn.copy()

    for i in range(data.shape[-1]):
        # turn out to be faster than (data-vS0)/vS1
        data[..., i] = (data[..., i]-vS[i, 0])/vS[i, 1]
    return data, vS
    

def transOut(dataIn, mtdLst, statIn):
    data = dataIn.copy()
    vS = statIn.copy()
    out = np.full(data.shape, np.nan)
    # turn out to be faster than (data-vS0)/vS1
    for i in range(data.shape[-1]):
        out[..., i] = data[..., i]*vS[i, 1]+vS[i, 0]
    # find colums that need to do log
    indLog = [i for i, mtd in enumerate(mtdLst) if mtd.split('-')[0] == 'log']
    out[..., indLog] = np.exp(out[..., indLog])-sn
    indLog2 = [i for i, mtd in enumerate(
        mtdLst) if mtd.split('-')[0] == 'log2']
    temp = out[..., indLog2].copy()
    temp[temp > 0] = np.exp(temp[temp > 0])-1
    temp[temp < 0] = -np.exp(-temp[temp < 0])+1
    out[..., indLog2] = temp
    return out
