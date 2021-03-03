from . import time
from . import stat
import numpy as np


def index2d(ind, ny, nx):
    iy = np.floor(ind / nx)
    ix = np.floor(ind % nx)
    return int(iy), int(ix)


def fillNan(mat, mask):
    temp = mat.copy()
    temp[~mask] = np.nan
    return temp


def flatData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    return xArray


def sortData(x):
    xArray = flatData(x)
    xSort = np.sort(xArray)
    return xSort


def rankData(x):
    ind = np.where(~np.isnan(x))[0]
    indS = np.argsort(x[ind])
    xRank = x[ind[indS]]
    return xRank, ind[indS]


def rmNan(xLst, returnInd=True):
    if len(set([len(x) for x in xLst])) > 1:
        raise Exception('not same size')
    n = len(xLst[0])
    ind = np.array(range(n))
    for x in xLst:
        indRm = np.where(np.isnan(x))
        ind = np.setdiff1d(ind, indRm)
    if returnInd is True:
        return [x[ind] for x in xLst], ind
    else:
        return [x[ind] for x in xLst]


def rmExt(data, p=5, returnInd=False):
    v1 = np.nanpercentile(data, p)
    v2 = np.nanpercentile(data, 100-p)
    ind = np.where((data > v1) & (data < v2))[0]
    out = data[ind]
    if returnInd:
        return out, ind
    else:
        return out

# class TimedOutExc(Exception):
#     pass

# def deadline(timeout, *args):
#     def decorate(f):
#         def handler(signum, frame):
#             raise TimedOutExc()

#         def new_f(*args):
#             signal.signal(signal.SIGALRM, handler)
#             signal.alarm(timeout)
#             return f(*args)
#             signal.alarm(0)

#         new_f.__name__ = f.__name__
#         return new_f
#     return decorate