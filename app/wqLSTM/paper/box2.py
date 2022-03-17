
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
# DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC


# LSTM
ep = 500
dataName = 'G200'
trainSet = 'rmR20'
testSet = 'pkR20'
# label = 'QFPRT2C'
label = 'FPRT2QC'

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
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# box plot
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})
dataPlot = list()
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codeLst]
for ic, code in enumerate(codeLst):
    dataPlot.append([corrL2[:, ic], corrW2[:, ic]])
    # dataPlot.append([corrL1[:, ic],corrL2[:, ic], corrW1[:, ic],corrW2[:, ic]])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=codeStrLst)
# fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(
#     12, 4), label1=codeStrLst, label2=['LSTM', 'WRTDS'])
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()
# dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
# plt.savefig(os.path.join(dirPaper, 'box_all'))
