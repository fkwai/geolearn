
import random
import scipy
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC


# LSTM corr
ep = 500
dataName = 'G200N'
trainSet = 'rmR20'
testSet = 'pkR20'
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
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL1, corrL2, corrW1, corrW2]:
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
DFN = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DFN, testSet=testSet, ep=500)
# deal with mean and std
codeLst = usgs.newC
yOut = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DFN.g[:, DFN.varG.index(code+'-M')]
    s = DFN.g[:, DFN.varG.index(code+'-S')]
    data = yP[:, :, k]
    yOut[:, :, k] = data*s+m
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# for each code
d1 = dbBasin.DataModelBasin(DFN, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DFN, subset=testSet, varY=codeLst)


code = '00618'
thR = 0.5
indC = codeLst.index(code)
ind1 = np.where(matLR[:, indC] <= thR)[0]
ind2 = np.where(matLR[:, indC] > thR)[0]


# ts map
lat, lon = DF.getGeo()
importlib.reload(figplot)


def funcM():
    figM, axM = plt.subplots(1, 1)
    axplot.mapPoint(axM, lat, lon, matLR[:, indC])
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP, axP = plt.subplots(1, 1, figsize=(15, 3))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    dataPlot = [yW[:, iP, indC], yOut[:, iP, indC],
                d1.Y[:, iP, indC], d2.Y[:, iP, indC]]
    cLst = ['blue', 'red', 'grey', 'black']
    axplot.plotTS(axP, DFN.t, dataPlot, cLst=cLst)
    titleStr = '{} {:.2f} {:.2f}'.format(
        DFN.siteNoLst[iP], corrL2[iP, indC], corrW2[iP, indC])
    axplot.titleInner(axP, titleStr)


figplot.clickMap(funcM, funcP)
