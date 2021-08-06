
import matplotlib.gridspec as gridspec
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
DFN = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.newC
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(
    outName, DF=DFN, testSet=testSet, ep=500)
yL = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DFN.g[:, DFN.varG.index(code+'-M')]
    s = DFN.g[:, DFN.varG.index(code+'-S')]
    yL[:, :, k] = yP[:, :, k]*s+m
siteNoLst = DFN.siteNoLst
ns = len(siteNoLst)
nc = len(codeLst)

# load WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# correlation matrix
d1 = dbBasin.DataModelBasin(DFN, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DFN, subset=testSet, varY=codeLst)
siteNoLst = DFN.siteNoLst
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
code = '00955'
indC = codeLst.index(code)

# 121
x = matW[:, indC, 3]
y = matL[:, indC, 3]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcM():
    x = matW[:, indC, 3]
    y = matL[:, indC, 3]
    figM = plt.figure(figsize=(12, 4))
    gsM = gridspec.GridSpec(1, 3)
    axM = list()
    axM.append(figM.add_subplot(gsM[0, 0]))
    axM.append(figM.add_subplot(gsM[0, 1:]))
    axM = np.array(axM)
    for k, v in enumerate(vLst):
        ind = np.where(tp[:, indC] == v)[0]
        axM[0].plot(x[ind], y[ind], c=cLst[k],
                    marker=mLst[k], ls='None')
    axplot.mapPointClass(axM[1], lat, lon, tp[:, indC], vLst=vLst, mLst=mLst,
                         cLst=cLst, labLst=labLst)
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    figM.suptitle(title)

    indV = np.where(tp[:, indC] != -1)[0]
    # xMat = np.stack([x[indV], lon[indV]], axis=1)
    # yMat = np.stack([y[indV], lat[indV]], axis=1)
    xMat = np.stack([x, lon], axis=1)
    yMat = np.stack([y, lat], axis=1)
    labelLst = ['scatter', 'map']
    axM[0].set_label('scatter')
    axM[1].set_label('map')

    figP = plt.figure(figsize=[12, 4])
    gsP = gridspec.GridSpec(2, 3)
    axP = list()
    axP.append(figP.add_subplot(gsP[0, :2]))
    axP.append(figP.add_subplot(gsP[1, :2]))
    axP.append(figP.add_subplot(gsP[:, 2]))
    axP = np.array(axP)
    return figM, axM, figP, axP, xMat, yMat, labelLst


def funcP(axP, iP, iM):
    print(iP, iM)
    indC = codeLst.index(code)
    Q = DF.q[:, iP, 1]
    C = DF.c[:, iP, indC]
    # TS-Q
    dataTS = [yL[:, iP, indC], yW[:, iP, indC],
              d1.Y[:, iP, indC], d2.Y[:, iP, indC]]
    cLst = ['r', 'b', 'grey', 'k']
    legLst = ['LSTM {:.2f}'.format(matL[iP, indC, 3]),
              'WRTDS {:.2f}'.format(matW[iP, indC, 3]),
              'training Obs', 'testing Obs']
    axplot.plotTS(axP[0], DF.t, dataTS, cLst=cLst, legLst=legLst)
    axplot.plotTS(axP[1], DF.t, [Q], cLst='b')
    # CQ
    a = matA[iP, indC, :]
    b = matB[iP, indC, :]
    p = matP[iP, indC, :]
    cqType.plotCQ(axP[2], Q, C, a, b, p)
    title = '{} {} {}'.format(
        siteNo, usgs.codePdf.loc[code]['shortName'], code)
    axP[2].set_title(title)


importlib.reload(figplot)
figM, figP = figplot.clickMulti(funcM, funcP)
