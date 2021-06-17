import importlib
from hydroDL.master import basins
from hydroDL.app.waterQuality import WRTDS
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
from hydroDL.model import rnn, crit, trainBasin, test
import torch
from torch import nn
import scipy

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

# dataName = 'weathering'
# sd = '1982-01-01'
# ed = '2018-12-31'
# dataName = 'weathering'
# freq = 'D'
# DF = dbBasin.DataFrameBasin.new(
#     dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)
dataName = 'weathering'
DF = dbBasin.DataFrameBasin(dataName)

indS = 0
q = DF.q[:, indS, 0]
c = DF.c[:, indS, :]
q1 = np.nanpercentile(q, 49)
q2 = np.nanpercentile(q, 51)
ind = np.where((q >= q1) & (q <= q2))[0]
c50 = np.nanmean(c[ind, :], axis=0)
# codeLst = DF.varC
codeLst = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
nfy = 3
nfx = 3
# hist of one site
bins = 10
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = DF.varC.index(code)
    ax.hist(c[:, ic], bins=bins)
    ax.axvline(x=c50[ic], color='r')
    nData = np.sum(~np.isnan(c[:, ic]))
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {} {}'.format(
        code, shortName, nData)
    axplot.titleInner(ax, titleStr)
fig.show()

# Q of one site
indS = 0
for indS in range(12):
    bins = 10
    fig, axes = plt.subplots(nfy, nfx)
    for k, code in enumerate(codeLst):
        j, i = utils.index2d(k, nfy, nfx)
        ax = axes[j, i]
        ic = DF.varC.index(code)
        q1 = np.log(DF.q[:, indS, 0]+1)
        ind = np.where(~np.isnan(DF.c[:, indS, ic]))[0]
        q2 = np.log(DF.q[ind, indS, 0]+1)
        axplot.plotCDF(ax, [q1, q2])
        # ax.hist([q1, q2], bins=bins, density=True)
        nData = np.sum(~np.isnan(DF.c[:, indS, ic]))
        shortName = usgs.codePdf.loc[code]['shortName']
        titleStr = '{} {} {}'.format(
            code, shortName, nData)
        axplot.titleInner(ax, titleStr)
    fig.show()

code = '00955'
siteNo = '01184000'
indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
c = DF.c[:, indS, indC]
q = DF.q[:, indS, 0]
q1 = np.nanpercentile(q, 49)
q2 = np.nanpercentile(q, 51)
ind = np.where((q >= q1) & (q <= q2))[0]
c50 = np.nanmean(c[ind], axis=0)
fig, ax = plt.subplots(1, 1)
ax.plot(np.log(q), c, '*')
ax.axhline(y=c50, color='r')
ax.axvline(x=np.log(q1), color='r')
ax.axvline(x=np.log(q2), color='r')
fig.show()


# temp = DF.c.reshape(-1, DF.c.shape[-1])
temp = DF.c.reshape(-1)
scipy.stats.normaltest(temp, nan_policy='omit')

scipy.stats.normaltest(DF.c[:, 0, :], nan_policy='omit')
