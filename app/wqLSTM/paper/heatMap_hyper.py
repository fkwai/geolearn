
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
DF = dbBasin.DataFrameBasin('G200')
ep = 500
codeLst = usgs.varC
trainSet = 'rmYr5'
testSet = 'pkYr5'

label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, 1000)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, 1000)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

hsLst = [16, 64, 128]

rhoLst = [180, 365, 750, 1000]
# count matrix

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)

corrLst1 = [corrL1]
corrLst2 = [corrL2]
caseLst = ['hs256-rho1000']
for hs in hsLst:
    outName = '{}-{}-{}-hs{}'.format(dataName, label, trainSet, hs)
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
    caseLst.append('hs{}'.format(hs))
for rho in rhoLst:
    outName = '{}-{}-{}-rho{}'.format(dataName, label, trainSet, rho)
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
    caseLst.append('rho{}'.format(rho))


# plot
figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
codeStrLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]

matPlot = np.full([len(corrLst2), len(codeLst)], np.nan)
for k, corr in enumerate(corrLst2):
    matPlot[k, :] = np.nanmean(corr, axis=0)
fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, matPlot*100, labLst=[caseLst, codeStrLst])
title = 'Median Testing Correlation'
ax.set_title(title)
# plt.tight_layout()
# fig.show()
# plt.savefig(os.path.join(
#     figFolder, 'heatmap_AllModel_{}'.format(trainSet)))
