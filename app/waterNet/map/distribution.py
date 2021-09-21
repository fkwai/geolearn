import random
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats


DF = dbBasin.DataFrameBasin('G200')

siteNoLst = DF.siteNoLst
codeLst = DF.varC
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)

matCount = np.zeros([len(siteNoLst), len(codeLst)])
for ic, code in enumerate(codeLst):
    temp = ~np.isnan(DF.q[:, :, 0]) & ~np.isnan(DF.c[:, :, ic])
    matCount[:, ic] = np.sum(temp, axis=0)


code = '00915'
indC = codeLst.index(code)
indS = np.where(matCount[:, indC] > 200)[0]
lat = dfCrd.loc[siteNoLst]['LAT_GAGE']
lon = dfCrd.loc[siteNoLst]['LNG_GAGE']
t = DF.t

C = DF.c[:, indS, indC]
Q = DF.q[:, indS, 1]/365*1000
P = DF.f[:, indS, DF.varF.index('pr')]
iP = 10

# fit gamma
iP = random.randint(0, len(indS))
[c], _ = utils.rmNan([C[:, iP]])
[q], _ = utils.rmNan([Q[:, iP]])

# fit c with gamma
data = c
pars = stats.gamma.fit(data)
rv = stats.gamma(pars[0], loc=pars[1], scale=pars[2])
stats.kstest(data, rv.cdf)
fig, ax = plt.subplots(1, 1)
_ = ax.hist(data, bins=30, density=True)
x = np.linspace(rv.ppf(0.01),
                rv.ppf(0.99), 100)
ax.plot(x, rv.pdf(x))
fig.show()

# fit q with lognorm
iP = random.randint(0, len(indS))
data = P[:, iP]
data = data[data > 1]

pars = stats.gamma.fit(data)
rv = stats.gamma(pars[0], loc=pars[1], scale=pars[2])
stats.kstest(data, rv.cdf)
fig, ax = plt.subplots(1, 1)
_ = ax.hist(data, bins=50, density=True)
x = np.linspace(rv.ppf(0.01),
                rv.ppf(0.99), 100)
ax.plot(x, rv.pdf(x))
fig.show()
stats.kstest(data, rv.cdf)


alpha = 5
loc = 100
beta = 22
rv = stats.gamma(alpha, loc=loc, scale=beta)
data = rv.rvs(size=10000)
stats.gamma.fit(data)
stats.kstest(data, rv.cdf)

fig, ax = plt.subplots(1, 1)
ax.hist(data, bins=20, density=True)
x = np.linspace(rv.ppf(0.01),
                rv.ppf(0.99), 100)
ax.plot(x, rv.pdf(x))
fig.show()
