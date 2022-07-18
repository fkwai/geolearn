
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

dataName = 'G200'
labelLst = ['FPRT2QC', 'QFPRT2C', 'QFRT2C', 'QFPT2C', 'QT2C']
trainSet = 'rmYr5'
testSet = 'pkYr5'

DF = dbBasin.DataFrameBasin('G200')
ep = 1000
codeLst = usgs.varC

# count matrix
matB = (~np.isnan(DF.c)).astype(int).astype(float)

# trainSet = 'rmR20'
# testSet = 'pkR20'
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
corrLst1 = list()
corrLst2 = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    outFolder = basinFull.nameFolder(outName)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    corr1 = np.load(corrFile1)
    corr1[matRm] = np.nan
    corrLst1.append(corr1)
    corr2 = np.load(corrFile2)
    corr2[matRm] = np.nan
    corrLst2.append(corr2)

# WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW1[matRm] = np.nan
corrLst1.append(corrW1)
corrW2 = np.load(corrFile2)
corrW2[matRm] = np.nan
corrLst2.append(corrW2)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
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


# re-order
indPlot = np.argsort(np.nanmean(matLR, axis=0))
codeStrLst = list()
dataPlot = list()
for k in indPlot:
    code = codeLst[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    tempLst = list()
    for kk in [1, 2, 3, 4, 0]:
        tempLst.append(corrLst2[kk][:, k])
    dataPlot.append(tempLst)
label2 = ['QFPV-C', 'QFV-C', 'QFP-C', 'Q-C', 'FPV-QC']
strLst = usgs.codeStrPlot(codeStrLst)

fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=strLst, label2=label2,
    cLst='kbgrc')
# fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(
#     12, 4), label1=codeStrLst, label2=['LSTM', 'WRTDS'])
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()
figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(figFolder, 'box_input_{}'.format(trainSet)))
fig.savefig(os.path.join(figFolder, 'box_input_{}.svg'.format(trainSet)))

indC = DF.varC.index('00945')
a=corrLst2[1][:,indC]
b=corrLst2[2][:,indC]
np.nanmean(a)
np.nanmean(b)
np.nanmedian(a)
np.nanmedian(b)