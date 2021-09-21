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

fig, ax = plt.subplots(1, 1)
ax.hist(C[:, iP], bins=20, density=True)
ax.hist(np.log(P[:, iP]+0.0001), bins=20, density=True)

fig.show()

# fit gamma
iP = random.randint(0, len(indS))
[c], _ = utils.rmNan([C[:, iP]])
c = c-np.mean(c)
stats.shapiro(c)
pars = stats.gamma.fit(c)
stats.kstest(c, 'gamma', pars)
fig, ax = plt.subplots(1, 1)
ax.hist(c, bins=20, density=True)
x = np.linspace(stats.gamma.ppf(0.01, pars),
                stats.gamma.ppf(0.99, pars), 100)
ax.plot(x, stats.gamma.pdf(x, pars))
fig.show()


gkde = stats.gaussian_kde(c)
ind = np.linspace(-7, 7, 101)
kdepdf = gkde.evaluate(ind)
fig, ax = plt.subplots(1, 1)
ax.hist(c, bins=10, normed=1)
ax.plot(ind, stats.norm.pdf(ind), color="r", label='DGP normal')
fig.show()
stats.shapiro(C[:, iP])


def funcM():
    figM, axM = plt.subplots(1, 1, figsize=(6, 4))
    axplot.mapPoint(axM, lat, lon, matCount[:, indC], s=16, cb=True)
    figP, axP1 = plt.subplots(1, 1, figsize=(12, 4))
    axP2 = axP1.twinx()
    axP = np.array([axP1, axP2])
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoLst[iP]
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    c = DF.c[:, iP, indC]
    q = DF.q[:, iP, 0]
    axP[1].plot(t, q, '-b')
    axP[0].plot(t, c, '*r')
    axP[0].xaxis_date()
    titleStr = '{} {} {} {}'.format(code, siteNo, area, matCount[iP, indC])
    axP[0].set_title(titleStr)
    print(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
