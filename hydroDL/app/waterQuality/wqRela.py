
from scipy.stats import linregress
import numpy as np
from hydroDL import utils
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from hydroDL.utils.stat import calPercent


def slopeModel(q, c, x=None):
    x1 = np.log(q)
    y1 = np.log(c)
    ind = np.where((~np.isnan(x1+y1)) & (~np.isinf(x1+y1)))
    a, b, r, p, std = linregress(x1[ind], y1[ind])
    sa = np.exp(b)
    sb = a
    if x is None:
        out = None
    else:
        out = sa*x**sb
    return sa, sb, out


def kateModel0(q, c, x=None):
    x2 = q
    y2 = 1/c
    ind = np.where((~np.isnan(x2+y2)) & (~np.isinf(x2+y2)))
    a, b, r, p, std = linregress(x2[ind], y2[ind])
    ceq = 1/b
    dw = 1/a/ceq
    if x is None:
        out = None
    else:
        out = ceq/(1+x/dw)
    return ceq, dw, out


def func(x, a, b):
    return a/(x/b+1)


def kateModel(q, c, x=None):
    (q, c), ind = utils.rmNan([q, c])
    popt, pcov = curve_fit(func, q, c, bounds=[
                           (0, 0), (np.inf, 100)])
    ceq = popt[0]
    dw = popt[1]
    if x is None:
        out = None
    else:
        out = ceq/(1+x/dw)
    return ceq, dw, out


def analRange(x, y, rIn, rOut, bIn=False, bOut=True):
    xIn = np.full([len(rIn)], np.nan)
    yIn = np.full([len(rIn)], np.nan)
    for j, r in enumerate(rIn):
        xIn[j] = calPercent(x, r, rank=bIn)
        yIn[j] = calPercent(y, r, rank=bIn)
    xOut = np.full([len(rIn)-1, len(rOut)], np.nan)
    yOut = np.full([len(rIn)-1, len(rOut)], np.nan)
    for j in range(len(rIn)-1):
        r1, r2 = (rIn[j], rIn[j+1])
        x1 = calPercent(x, r1, rank=bIn)
        x2 = calPercent(x, r2, rank=bIn)
        indX = np.where((x > x1) & (x <= x2))[0]
        yy = y[indX]
        for i in range(len(rOut)):
            yOut[j, i] = calPercent(yy, rOut[i], rank=bOut)
        y1 = calPercent(y, r1, rank=bIn)
        y2 = calPercent(y, r2, rank=bIn)
        indY = np.where((y > y1) & (y <= y2))[0]
        xx = x[indY]
        for i in range(len(rOut)):
            xOut[j, i] = calPercent(xx, rOut[i], rank=bOut)
    return xIn, yIn, xOut, yOut
