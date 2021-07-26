import numpy as np
from hydroDL import kPath, utils
import statsmodels.api as sm
import matplotlib.pyplot as plt
"""
Repeat Moatar2016
"""
# calculate C50
sn = 1e-5
vLst = [0, 1, 2, 3, 4, 5, 6, 7]
labLst = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
cLst = np.array([
    [0.5, 0.5, 0.5, 1],  # 00
    [1, 0.5, 0.5, 1],  # 01
    [0.5, 0.5, 1, 1],  # 02
    [0.5, 0.5, 0, 1],  # 10
    [1, 0, 0, 1],  # 11
    [1, 0.5, 1, 1],  # 12
    [0, 0.2, 0.5, 1],  # 20
    [1, 0, 1, 1],  # 21
    [0, 0, 1, 1]  # 22
])
mLst = 'P>><PX<XP'
# cLst = plt.cm.jet(np.linspace(0, 1, 9))


def getPlotArg():
    return vLst, cLst,  mLst, labLst


def calPar(Q, C):
    q = np.log(Q+sn)
    c = np.log(C+sn)
    q1 = np.nanmedian(q)
    ind1 = np.where(q <= q1)[0]
    ind2 = np.where(q > q1)[0]
    a, b, p = [np.array([np.nan, np.nan]) for k in range(3)]
    for k, ind in enumerate([ind1, ind2]):
        [x, y], _ = utils.rmNan([q[ind], c[ind]])
        if len(x) > 10:
            mod = sm.OLS(y, sm.add_constant(x))
            res = mod.fit()
            a[k] = res.params[0]
            b[k] = res.params[1]
            p[k] = res.pvalues[1]
    return a, b, p


def plotCQ(ax, Q, C, a, b, p):
    q = np.log(Q+sn)
    c = np.log(C+sn)
    q1 = np.nanmedian(q)
    c1 = q1*b+a
    q0 = np.nanmin(q)
    c0 = q0*b[0]+a[0]
    q2 = np.nanmax(q)
    c2 = q2*b[1]+a[1]
    [c0, c2] = [q0, q2]*b+a
    [x, y], _ = utils.rmNan([q, c])
    ax.plot(x, y, 'k*')
    ax.plot([q0, q1], [c0, c1[0]], 'r-',
            label='{:.2f} {:.0e}'.format(b[0], p[0]))
    ax.plot([q1, q2], [c1[1], c2], 'r-',
            label='{:.2f} {:.0e}'.format(b[1], p[1]))
    ax.legend()
    return ax


def par2type(matB, matP, th=0.05):
    th = 0.05
    h1 = (matP[:, :, 0] < th).astype(int)
    s1 = (matB[:, :, 0] > 0).astype(int)+1
    h2 = (matP[:, :, 1] < th).astype(int)
    s2 = (matB[:, :, 1] > 0).astype(int)+1
    tp = h1*s1*3+h2*s2
    tp[np.any(np.isnan(matP), axis=2)] = -1
    return tp


def ternary(n):
    e = n//3
    q = n % 3
    if n == 0:
        return '0'
    elif e == 0:
        return str(q)
    else:
        return ternary(e) + str(q)
