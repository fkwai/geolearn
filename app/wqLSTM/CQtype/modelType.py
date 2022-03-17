
from mpl_toolkits import basemap
from numpy.core.function_base import linspace
import pandas as pd
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time
from hydroDL.master import basinFull

# load models
dataName = 'G200N'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.newC
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(
    outName, DF=DF, testSet=testSet, ep=500)
yL = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DF.g[:, DF.varG.index(code+'-M')]
    s = DF.g[:, DF.varG.index(code+'-S')]
    yL[:, :, k] = yP[:, :, k]*s+m
siteNoLst = DF.siteNoLst
ns = len(siteNoLst)
nc = len(codeLst)

# load WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
matW = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
matL = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            statL = utils.stat.calStat(yL[:, indS, indC], d2.Y[:, indS, indC])
            matL[indS, indC, :] = list(statL.values())

# load pars
filePar = os.path.join(kPath.dirWQ, 'modelStat', 'typeCQ', dataName+'.npz')
npz = np.load(filePar)
matA = npz['matA']
matB = npz['matB']
matP = npz['matP']

# get types
importlib.reload(axplot)
importlib.reload(cqType)
tp = cqType.par2type(matB, matP)
vLst, cLst,  mLst, labLst = cqType.getPlotArg()

# scatter corr
nfy, nfx = [4, 5]
fig, axes = plt.subplots(nfy, nfx)
ticks = [0, 0.5, 1]
for kk, code in enumerate(codeLst):
    j, i = utils.index2d(kk, nfy, nfx)
    ax = axes[j, i]
    indC = codeLst.index(code)
    x = matW[:, indC, 3]
    y = matL[:, indC, 3]
    for k, v in enumerate(vLst):
        ind = np.where(tp[:, indC] == v)[0]
        ax.plot(x[ind], y[ind], c=cLst[k],
                marker=mLst[k], ls='None')
    ax.plot([0, 1], [0, 1], '-k')
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    axplot.titleInner(ax, title)
    # change ticks
    _ = ax.set_xlim([ticks[0], ticks[-1]])
    _ = ax.set_ylim([ticks[0], ticks[-1]])
    _ = ax.set_yticks(ticks)
    _ = ax.set_xticks(ticks)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != nfy:
        _ = ax.set_xticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
fig.show()

# scatter bias
nfy, nfx = [4, 5]
fig, axes = plt.subplots(nfy, nfx)
for kk, code in enumerate(codeLst):
    j, i = utils.index2d(kk, nfy, nfx)
    ax = axes[j, i]
    indC = codeLst.index(code)
    x = matW[:, indC, 1]
    y = matL[:, indC, 1]
    vR = utils.vRange(np.concatenate([x, y]), prct=99)
    # ticks = [vR[0], 0, vR[1]]
    ticks = np.linspace(vR[0],  vR[1], 3)
    for k, v in enumerate(vLst):
        ind = np.where(tp[:, indC] == v)[0]
        ax.plot(x[ind], y[ind], c=cLst[k],
                marker=mLst[k], ls='None')
    ax.plot([vR[0], vR[1]], [vR[0], vR[1]], '-k')
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    axplot.titleInner(ax, title)
    _ = ax.set_xlim([ticks[0], ticks[-1]])
    _ = ax.set_ylim([ticks[0], ticks[-1]])
    _ = ax.set_yticks(ticks)
    _ = ax.set_xticks(ticks)
fig.show()

# create a legend
fig, ax = plt.subplots(1, 1)
for k in range(9):
    ax.plot(0, 0, c=cLst[k], label=labLst[k], marker=mLst[k], ls='None',
            markersize=20)
ax.legend(fontsize=20)
fig.show()
