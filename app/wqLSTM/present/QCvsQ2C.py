
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
trainSet = 'rmR20'
testSet = 'pkR20'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = DF.varC
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
bq1 = DF.extractSubset(np.isnan(DF.q[:, :, 0:1]), trainSet)[:, :, 0]
bq2 = DF.extractSubset(np.isnan(DF.q[:, :, 0:1]), testSet)[:, :, 0]


# LSTM
# labelLst = ['QT2C', 'FPRT2QC']
labelLst = ['QFPRT2C', 'FPRT2QC']
yPLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=500)
    master = basinFull.loadMaster(outName)
    indC = [master['varY'].index(x) for x in codeLst]
    yP = yP[:, :, indC]
    yPLst.append(yP)

# correlation
corrLst1 = list()
corrLst2 = list()
bQ = np.isnan(DF.q[:, :, 0])
for yP in yPLst:
    yT = yP.copy()
    yT[bQ, :] = np.nan
    corr1 = utils.stat.calCorr(DF.extractSubset(yT, trainSet), obs1)
    corr2 = utils.stat.calCorr(DF.extractSubset(yT, testSet), obs2)
    corrLst1.append(corr1)
    corrLst2.append(corr2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count = np.nansum(matB1, axis=0)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 160) & (count2 < 40)

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values

code = '00945'
ic = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
ind = np.where(count[:, ic] > 100)[0]
ax.scatter(corrLst2[0][ind, ic], corrLst2[1][ind, ic],
           c=matLR[ind, ic])
ax.plot([0, 1], [0, 1], '-k')
fig.show()


dataPlot = list()
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codeLst]
for ic, code in enumerate(codeLst):
    a = corrLst2[0][:, ic]
    b = corrLst2[1][:, ic]
    bCount = (count1[:, ic] > 160) & (count2[:, ic] > 40)
    ind1 = np.where((a > b) & bCount)[0]
    ind2 = np.where((a < b) & bCount)[0]
    dataPlot.append([matLR[ind1, ic], matLR[ind2, ic]])
    # dataPlot.append([c1, c2])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=codeStrLst)
fig.show()
