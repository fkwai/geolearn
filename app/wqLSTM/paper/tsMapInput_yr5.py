
import matplotlib.dates as mdates
import random
import scipy
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


# LSTM corr
ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
labelLst = ['QFPRT2C', 'QFRT2C']
corrLst1 = list()
corrLst2 = list()
outNameLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    outFolder = basinFull.nameFolder(outName)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    corrL1 = np.load(corrFile1)
    corrL2 = np.load(corrFile2)
    corrLst1.append(corrL1)
    corrLst2.append(corrL2)
    outNameLst.append(outName)


# WRTDS corr
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW2 = np.load(corrFile2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrW1, corrW2]+corrLst1+corrLst2:
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

# load TS
yP1, ycP = basinFull.testModel(outNameLst[0], DF=DF, testSet=testSet, ep=1000)
yP2, ycP = basinFull.testModel(outNameLst[1], DF=DF, testSet=testSet, ep=1000)
codeLst = usgs.varC

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# SO4 load
indF = DF.varF.index('SO4')
matP = np.nanmean(DF.f[:, :, indF], axis=0)
indF2 = DF.varF.index('distNTN')
matP2 = np.nanmean(DF.f[:, :, indF2], axis=0)
code = '00945'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
x = corrLst2[0][indS, indC]**2 - corrLst2[1][indS, indC]**2
fig, ax = plt.subplots(1, 1)
# ax.scatter(x, matP[indS], c=matP2[indS])
ax.plot(matP[indS],x,'*')
fig.show()

# ts map
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

lat, lon = DF.getGeo()
code = '00945'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
importlib.reload(figplot)
importlib.reload(axplot)
yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)


xMat = np.ndarray([len(indS), 3])
yMat = np.ndarray([len(indS), 3])
xMat[:, 0] = corrLst2[0][indS, indC]
yMat[:, 0] = corrLst2[1][indS, indC]
xMat[:, 1] = lon[indS]
yMat[:, 1] = lat[indS]


def funcM():
    figM = plt.figure(figsize=(14, 3))
    gsM = gridspec.GridSpec(1, 3)
    labelLst = ['scatter', 'map']
    axS = figM.add_subplot(gsM[0, :1])
    axS.set_label(labelLst[0])
    cs = axplot.scatter121(axS, xMat[:, 0], yMat[:, 0], matP[indS])
    plt.colorbar(cs, orientation='vertical')
    dataMap = corrLst2[0][indS, indC]**2 - corrLst2[1][indS, indC]**2
    axM1 = mapplot.mapPoint(
        figM, gsM[0, 1:3], lat[indS], lon[indS], dataMap)
    axM1.set_label(labelLst[1])
    axM = np.array([axS, axM1])
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, ny, wspace=0)
    axP0 = figP.add_subplot(gsP[0, 0])
    axPLst = [axP0]
    for k in range(1, ny):
        axP = figP.add_subplot(gsP[0, k], sharey=axP0)
        axPLst.append(axP)
    axP = np.array(axPLst)
    return figM, axM, figP, axP, xMat, yMat, labelLst


def funcP(axP, iP, iM):
    print(iP, iM)
    k = indS[iP]
    dataPlot = [yW[:, k, indC], yP1[:, k, indC], yP2[:, k, indC],
                DF.c[:, k, DF.varC.index(code)]]
    cLst = 'kbgr'
    legLst = ['WRTDS', 'LSTM1', 'LSTM2', 'Obs']
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst,
                     legLst=legLst, styLst='---*')
    titleStr = '{} {:.2f} {:.2f} {:.2f}'.format(
        DF.siteNoLst[k], corrLst2[0][k, indC], corrLst2[1][k, indC],
        corrW2[k, indC])
    print(titleStr)


figM, figP = figplot.clickMulti(funcM, funcP)

for k in range(len(codeLst)):
    a = np.nanmean(corrLst2[0][:, k], axis=0)
    b = np.nanmean(corrLst2[1][:, k], axis=0)
    print(usgs.codePdf.loc[codeLst[k]]['shortName'], a/b)
