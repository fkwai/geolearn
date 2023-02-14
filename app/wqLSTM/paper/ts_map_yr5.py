
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
ep = 500
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corr-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corr-{}-Ep{}.npy'.format(testSet, ep)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

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
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# load linear/seasonal
# dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
dirPar = os.path.join(kPath.dirWQ,'modelStat','LR-All','QS','param')
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

# load TS
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=500)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# ts map
lat, lon = DF.getGeo()
code = '00660'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
importlib.reload(figplot)
importlib.reload(axplot)
yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat[indS], lon[indS], matLR[indS, indC])
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, ny, wspace=0)
    axP0 = figP.add_subplot(gsP[0, 0])
    axPLst = [axP0]
    for k in range(1, ny):
        axP = figP.add_subplot(gsP[0, k], sharey=axP0)
        axPLst.append(axP)
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon[indS], lat[indS]


def funcP(iP, axP):
    print(iP)
    k = indS[iP]
    dataPlot = [yW[:, k, indC], yP[:, k, indC],
                DF.c[:, k, DF.varC.index(code)]]
    cLst = 'kbr'
    legLst = ['WRTDS', 'LSTM', 'Obs']
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst, legLst=legLst)
    titleStr = '{} {:.2f} {:.2f}'.format(
        DF.siteNoLst[k], corrL2[k, indC], corrW2[k, indC])
    print(titleStr)


figplot.clickMap(funcM, funcP)

