
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
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
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
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
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
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=1000)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# ts map
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

lat, lon = DF.getGeo()
code = '00660'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
importlib.reload(figplot)
importlib.reload(axplot)
yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)

xMat = np.ndarray([len(indS), 3])
yMat = np.ndarray([len(indS), 3])
xMat[:, 0] = corrL2[indS, indC]
yMat[:, 0] = corrW2[indS, indC]
xMat[:, 1] = lon[indS]
yMat[:, 1] = lat[indS]
xMat[:, 2] = lon[indS]
yMat[:, 2] = lat[indS]


def funcM():
    figM = plt.figure(figsize=(14, 3))
    gsM = gridspec.GridSpec(1, 5)
    labelLst = ['scatter', 'map1', 'map2']
    axS = figM.add_subplot(gsM[0, :1])
    axS.set_label(labelLst[0])
    cs = axplot.scatter121(axS, xMat[:, 0], yMat[:, 0], matLR[indS, indC])
    plt.colorbar(cs, orientation='vertical')

    axM1 = mapplot.mapPoint(
        figM, gsM[0, 1:3], lat[indS], lon[indS], corrL2[indS, indC])
    axM1.set_label(labelLst[1])
    axM2 = mapplot.mapPoint(
        figM, gsM[0, 3:], lat[indS], lon[indS],
        corrL2[indS, indC]**2-corrW2[indS, indC]**2)
    axM2.set_label(labelLst[2])
    axM = np.array([axS, axM1, axM2])
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
    dataPlot = [yW[:, k, indC], yP[:, k, indC],
                DF.c[:, k, DF.varC.index(code)]]
    cLst = 'kbr'
    legLst = ['WRTDS', 'LSTM', 'Obs']
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst, legLst=legLst)
    titleStr = '{} {:.2f} {:.2f}'.format(
        DF.siteNoLst[k], corrL2[k, indC], corrW2[k, indC])
    print(titleStr)


figM, figP = figplot.clickMulti(funcM, funcP)

# overall stats
a = matLR[indS, indC]
b = corrW2[indS, indC]
c = corrL2[indS, indC]
ind1 = np.where(b > c)[0]
ind2 = np.where(b < c)[0]
np.median(a[ind1])
np.median(a[ind2])
len(ind1)
len(ind2)

the = np.percentile(a, 40)
ind1 = np.where(a < the)[0]
ind2 = np.where(a > the)[0]
np.median(b[ind1])
np.median(c[ind1])
np.median(b[ind2])
np.median(c[ind2])
np.nanmedian(matLR, axis=0)

np.median(b)
np.median(c)
np.mean(b)
np.mean(c)

# for a specfic site
# code = '00665'
# siteNo = '01631000'
# indS = DF.siteNoLst.index(siteNo)
# # extract 5 yr
# indLst = list()
# ny = len(yrLst)
# ty = DF.t.astype('M8[Y]').astype(str).astype(int)
# for yr in yrLst:
#     bp = np.in1d(ty, yr)
#     ind = np.where(bp)[0]
#     indLst.append(ind)
# indAry = np.concatenate(indLst)
# a = yW[:, indS, indC]
# b = yP[:, indS, indC]
# c = DF.c[:, indS, indC]
# aY = a[indAry]
# bY = b[indAry]
# cY = c[indAry]
# [aa, bb, cc], indT = utils.rmNan([aY, bY, cY])
# np.corrcoef(aa, cc)
# np.corrcoef(bb, cc)
# fig, ax = plt.subplots(1, 1)
# ax.plot(cc, aa, 'b*')
# ax.plot(cc, bb, 'r*')
# fig.show()
# fig, ax = plt.subplots(1, 1)
# ax.plot(cc, aa, 'b*')
# ax.plot(cc, bb, 'r*')
# fig.show()