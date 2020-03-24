
from scipy.stats import linregress
import numpy as np


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


def kateModel(q, c, x=None):
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
