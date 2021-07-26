
from mpl_toolkits import basemap
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

# plot for code
code = '00300'
indC = codeLst.index(code)

# 121
x = matW[:, indC, 3]
y = matL[:, indC, 3]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for k, v in enumerate(vLst):
    ind = np.where(tp[:, indC] == v)[0]
    axes[0].plot(x[ind], y[ind], c=cLst[k],
                 marker=mLst[k], ls='None')
axes[0].set_title('types')
c = matB[:, indC, 0]
vR = utils.vRange(c)
axes[1].scatter(x, y, c=c, vmin=vR[0], vmax=vR[1], cmap='jet')
axes[1].set_title('slope < C50')
c = matB[:, indC, 1]
vR = utils.vRange(c)
axes[2].scatter(x, y, c=c, vmin=vR[0], vmax=vR[1], cmap='jet')
axes[2].set_title('slope > C50')
for ax in axes:
    ax.plot([0, 1], [0, 1], '-k')
title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
fig.suptitle(title)
fig.show()


# box plot of types
dataBox = list()
labelLst1 = list()
for k, v in enumerate(vLst):
    ind = np.where(tp[:, indC] == v)[0]
    if len(ind) > 5:
        dataBox.append([matL[ind, indC, 3], matW[ind, indC, 3]])
        labelLst1.append(labLst[k])

fig, axes = figplot.boxPlot(dataBox, widths=0.5, figsize=(12, 4),
                            label2=['LSTM', 'WRTDS'], label1=labelLst1)
fig.show()
