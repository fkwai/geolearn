import datetime as dt
from hydroDL import utils

def t2yrLst(tArray):
    t1 = tArray[0].astype(object)
    t2 = tArray[-1].astype(object)
    y1 = t1.year
    y2 = t2.year
    if t1 < dt.date(y1, 4, 1):
        y1 = y1 - 1
    if t2 < dt.date(y2, 4, 1):
        y2 = y2 - 1
    yrLst = list(range(y1, y2 + 1))
    tDb = utils.time.tRange2Array([dt.date(y1, 4, 1), dt.date(y2 + 1, 4, 1)])
    return yrLst, tDb
