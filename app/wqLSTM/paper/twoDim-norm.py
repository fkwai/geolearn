
import matplotlib
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
from hydroDL.post import adjustText
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC

# LSTM
ep = 500
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
dataNameLst = ['G200', 'G200N']
corrLst1 = list()
corrLst2 = list()
for dataName in dataNameLst:
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


# WRTDS
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
matRm = (count1 < 80) & (count2 < 20)
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


##
var = DF.varC.copy()
var.remove('00400')
iC = np.array([DF.varC.index(code) for code in var])
iR = np.argsort(np.nanmedian(matLR[:, iC], axis=0))
ind = iC[iR]
varP = [DF.varC[k] for k in ind]
x = np.nanmedian(matLR[:, ind], axis=0)
y1 = np.nanmedian(corrW2[:, ind]**2, axis=0)
y2 = np.nanmedian(corrLst2[0][:, ind]**2, axis=0)
y3 = np.nanmedian(corrLst2[1][:, ind]**2, axis=0)


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(x, y1, 'b-*', label='WRTDS')
ax.plot(x, y2, 'r-*', label='LSTM')
ax.fill_between(
    x, y1, y2, where=y1 > y2, facecolor='blue', alpha=0.5,
    interpolate=True, label='LSTM < WRTDS')
ax.fill_between(
    x, y1, y2, where=y2 >= y1, facecolor='red', alpha=0.5,
    interpolate=True, label='LSTM > WRTDS')
ax.plot(x, y3, 'g--', label='LSTM local-norm')

txtLst = list()
for k, code in enumerate(varP):
    codeStr = usgs.codePdf.loc[code]['shortName']
    if codeStr in usgs.dictLabel.keys():
        txt = ax.text(x[k], y1[k], usgs.dictLabel[codeStr], ha='center')
    else:
        txt = ax.text(x[k], y1[k], codeStr, ha='center')
    txtLst.append(txt)
# adjustText.adjust_text(txtLst)
ax.legend()
# ax.set_xlabel('simplicity')
# ax.set_ylabel('testing Rsq')
fig.show()
figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(figFolder, 'twoDim'.format(label, trainSet)))
fig.savefig(os.path.join(figFolder, 'twoDim.svg'.format(label, trainSet)))
