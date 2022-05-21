
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

dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
# label = 'FPRT2QC'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1500)
codeLst = usgs.varC
if yP.shape[2] > len(codeLst):
    yP = yP[:, :, 1:]


# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
corrL1 = utils.stat.calCorr(DF.extractSubset(yP, trainSet), obs1)
corrL2 = utils.stat.calCorr(DF.extractSubset(yP, testSet), obs2)
corrW1 = utils.stat.calCorr(DF.extractSubset(yW, trainSet), obs1)
corrW2 = utils.stat.calCorr(DF.extractSubset(yW, testSet), obs2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# box plot
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})

# re-order
indPlot = np.argsort(np.nanmean(corrW2, axis=0))
codeStrLst = list()
dataPlot = list()
for k in indPlot:
    code = codeLst[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([corrL2[:, k], corrW2[:, k]])

fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=codeStrLst)
# fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(
#     12, 4), label1=codeStrLst, label2=['LSTM', 'WRTDS'])
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
plt.savefig(os.path.join(dirPaper, 'box_all'))
