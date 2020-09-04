
from scipy.stats import linregress
import numpy as np
from hydroDL import utils
from scipy.optimize import curve_fit


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
