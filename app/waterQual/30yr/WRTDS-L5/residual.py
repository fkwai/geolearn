import scipy
import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

wqData = waterQuality.DataModelWQ('rbWN5')
siteNoLst = wqData.siteNoLst
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'All')
dirOut = os.path.join(dirWRTDS, 'output')
dirPar = os.path.join(dirWRTDS, 'params')

# read a temp file
saveFile = os.path.join(dirOut, siteNoLst[0])
dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
t = dfP.index
nt = len(dfP.index)
nc = len(usgs.newC)
ns = len(siteNoLst)
matR = np.ndarray([ns, nt, nc])
matC = np.ndarray([ns, nt, nc])

# calculate residual
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveFile = os.path.join(dirOut, siteNo)
    dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
    dfP.index = pd.to_datetime(dfP.index)
    dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.newC, freq='W')
    matR[kk, :, :] = dfP.values-dfC.values
    matC[kk, :, :] = dfC.values


codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']

# plot hist
importlib.reload(axplot)
importlib.reload(transform)
importlib.reload(usgs)

varRLst = [code+'-R' for code in usgs.newC]
mtdLst = waterQuality.extractVarMtd(varRLst)
matRN, stat = transform.transInAll(matR, mtdLst)
matRN2 = transform.transOutAll(matRN, mtdLst, stat)

fig, axes = plt.subplots(5, 4)
ticks = [-0.5, 0, 0.5, 1]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    ic = usgs.newC.index(code)
    data = matRN2[indS, :, ic]
    x1 = utils.flatData(data)
    x2 = utils.rmExt(x1, p=5)

    s, p = scipy.stats.kstest(x2/np.std(x2)-np.mean(x2), 'laplace')
    # s, p = scipy.stats.shapiro(x2)
    # s, p = scipy.stats.shapiro(x2)
    # scipy.stats.anderson(x2, dist='norm')
    _ = ax.hist(x2, bins=200, density=True)
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {} s={:.3f} p={:.3f}'.format(code, shortName, s, p)
    axplot.titleInner(ax, titleStr)
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.colorbar()
fig.show()

# test for different dist
distLst = ['logistic', 'cauchy', 'norm']
cLst = 'rgbm'
importlib.reload(axplot)
fig, axes = plt.subplots(5, 4)
ticks = [-0.5, 0, 0.5, 1]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    ic = usgs.newC.index(code)
    data = matR[indS, :, ic]
    x1 = utils.flatData(data)
    x2 = utils.rmExt(x1, p=5)
    _ = ax.hist(x2, bins=100, density=True, color='grey')
    for j, distName in enumerate(distLst):
        dist = getattr(scipy.stats, distName)
        args = dist.fit(x2)
        s, p = scipy.stats.kstest(x2, distName, args=args)
        xx = np.sort(x2)
        yy = dist(*args).pdf(xx)
        if k == 0:
            label = '{} {:.2f}'.format(distName, p)
        else:
            label = '{:.2f}'.format(p)
        _ = ax.plot(xx, yy, color=cLst[j], label=label)
    _ = ax.legend()
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {}'.format(code, shortName)
    axplot.titleInner(ax, titleStr)
fig.show()

# plot mean vs std for each site
importlib.reload(axplot)
fig, axes = plt.subplots(5, 4)
ticks = [-0.5, 0, 0.5, 1]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    ic = usgs.newC.index(code)
    mean = np.nanstd(matC[indS, :, ic], axis=1)
    std = np.nanmean(matR[indS, :, ic], axis=1)
    # mean = np.ndarray(len(indS))
    # std = np.ndarray(len(indS))
    # for k in range(len(indS)):
    #     x = data[k, :]
    #     xx = utils.rmExt(x)
    #     mean[k] = np.nanmean(xx)
    #     std[k] = np.nanstd(xx)
    # _ = axplot.plot121(ax, mean, std)
    ax.plot(mean, std, '*b')
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {}'.format(code, shortName)
    axplot.titleInner(ax, titleStr)
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.colorbar()
fig.show()

code = '00915'
indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
fig, ax = plt.subplots(1, 1)
x = matC[indS[2], :, ic]
y = matR[indS[2], :, ic]
ax.plot(x, y, '*b')

fig.show()

#
code = '00600'
distName = 'cauchy'
fig, ax = plt.subplots(1, 1)
siteNoCode = dictSite[code]
indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
ic = usgs.newC.index(code)
data = matR[indS, :, ic]
x1 = utils.flatData(data)
x2 = utils.rmExt(x1, p=5)
dist = getattr(scipy.stats, distName)
args = dist.fit(x2)
s, p = scipy.stats.kstest(x2, distName, args=args)
xx = np.sort(x2)
yy = dist(*args).pdf(xx)
_ = ax.hist(x2, bins=200, density=True)
_ = ax.plot(xx, yy, '-r')
fig.show()
