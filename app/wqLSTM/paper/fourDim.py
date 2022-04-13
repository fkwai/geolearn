
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
codeLst = usgs.varC


# LSTM
ep = 500
dataName = 'G200'
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

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

#
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
a = np.nanmean(matLR, axis=0)
b = np.nanmean(corrL2**2 - corrW2**2, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
for k in range(len(codeLst)):
    ax.text(a[k], b[k], usgs.codePdf.loc[codeLst[k]]['shortName'], fontsize=16)
ax.plot(a, b, '*')
ax.axhline(0, color='r')
ax.axvline(0.4, color='r')
ax.set_xlabel('Simplicity of Variable')
ax.set_ylabel('LSTM Rsq minus WRTDS Rsq')
fig.show()
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
# plt.savefig(os.path.join(dirPaper, 'fourDim'))

#
a = np.nanmean(matLR, axis=0)
b = np.nanmean(corrL2**2, axis=0)
c = np.nanmean(corrW2**2, axis=0)

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(a[k], (b[k]+c[k])/2, usgs.codePdf.loc[codeLst[k]]['shortName'])
    ax.plot([a[k], a[k]], [b[k], c[k]], c='0.5')
ax.plot(a, b, 'r*')
ax.plot(a, c, 'b*')
# ax.set_xscale('log')

fig.show()
