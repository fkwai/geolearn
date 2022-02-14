
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

# LSTM

# LSTM
# labelLst = ['QT2C', 'FPRT2QC']
labelLst = ['FPRT2QC', 'QT2C', 'QFPRT2C']
yPLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=500)
    master = basinFull.loadMaster(outName)
    indC = [master['varY'].index(x) for x in codeLst]
    yP = yP[:, :, indC]
    yPLst.append(yP)

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values

# correlation
corrLst1 = list()
bQ = np.isnan(DF.q[:, :, 0])
for yP in yPLst:
    yT = yP.copy()
    yT[bQ, :] = np.nan
    corr = utils.stat.calCorr(DF.extractSubset(yT, testSet), obs2)
    corrLst1.append(corr)
corrLst2 = list()
bQ = np.isnan(DF.q[:, :, 0])
for yP in yPLst:
    yT = yP.copy()
    yT[bQ, :] = np.nan
    corr = utils.stat.calCorr(DF.extractSubset(yT, testSet),
                              DF.extractSubset(yW, testSet))
    corrLst2.append(corr)
corrW = utils.stat.calCorr(DF.extractSubset(yW, testSet), obs2)
# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 160) & (count2 < 40)
for corr in corrLst1:
    corr[matRm] = np.nan
for corr in corrLst2:
    corr[matRm] = np.nan


# box plot - all cases
dataPlot = list()
# codePlot = [codeLst[k] for k in np.argsort(np.nanmean(matLR, axis=0))]
codePlot = ['00935', '00955', '00940', '00945',
            '00930', '00095', '00915', '00925']
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codePlot]
for code in codePlot:
    ic = codeLst.index(code)
    # dataPlot.append([corr[:, ic] for corr in corrLst1]+[corrW[:, ic]])
    dataPlot.append([corr[:, ic] for corr in corrLst1])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label1=codeStrLst, cLst='rgbk',
                            label2=['F-QC', 'Q-C', 'FQ-C'])
fig.show()

# box plot
dataPlot = list()
codePlot = [codeLst[k] for k in np.argsort(np.nanmean(matLR, axis=0))]
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codePlot]
for code in codePlot:
    ic = codeLst.index(code)
    dataPlot.append([corrLst1[2][:, ic], corrW[:, ic]])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label1=codeStrLst, cLst='rb', yRange=[0, 1],
                            label2=['LSTM', 'WRTDS'])
fig.show()

# walk through complexity
a = np.nanmean(matLR, axis=0)
b = np.nanmedian(corrLst1[0]**2, axis=0)
c = np.nanmedian(corrLst1[1]**2, axis=0)
d = np.nanmean(corrLst1[2]**2, axis=0)
e = np.nanmean(corrW**2, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
for k in range(len(codeLst)):
    ax.text(a[k], b[k], usgs.codePdf.loc[codeLst[k]]['shortName'], fontsize=16)
ax.plot(a, b, 'b*')
ax.plot(a, c, 'r*')
# ax.plot(a, d, 'r*')
# ax.plot(a, e, 'k*')
# ax.axhline(0, color='r')
# ax.axvline(0.4, color='r')
ax.set_xlabel('Simplicity of Variable')
ax.set_ylabel('LSTM Rsq minus WRTDS Rsq')
ax.set_xscale('log')
ax.set_yscale('log')
fig.show()

code = '00600'
ic = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
ax.scatter(corrLst1[0][:, ic], corrLst1[1][:, ic], c=matLR[:, ic])
ax.plot([0, 1], [0, 1], '-k')
fig.show()
