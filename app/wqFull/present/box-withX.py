
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
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=500)
master = basinFull.loadMaster(outName)
indC = [master['varY'].index(x) for x in codeLst]
yP = yP[:, :, indC]

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
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


def width(p, w):
    return 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)


pos = np.nanmean(matLR, axis=0)

codePlot = [codeLst[k] for k in np.argsort(np.nanmean(matLR, axis=0))]
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codePlot]
fig, ax = plt.subplots(1, 1)
dataPlot1 = list()
dataPlot2 = list()
for ic, code in enumerate(codeLst):
    tempL = corrL2[:, ic]
    tempW = corrW2[:, ic]
    dataPlot1.append(tempL[~np.isnan(tempL)])
    dataPlot2.append(tempW[~np.isnan(tempW)])
w = 0.01

bp1 = ax.boxplot(dataPlot1, widths=width(pos, w),
                 positions=pos-width(pos, w)/2,
                 patch_artist=True, notch=True, showfliers=False)
for kk in range(0, len(bp1['boxes'])):
    plt.setp(bp1['boxes'][kk], facecolor='r')
bp2 = ax.boxplot(dataPlot2, widths=width(pos, w),
                 positions=pos+width(pos, w)/2,
                 patch_artist=True, notch=True, showfliers=False)
for kk in range(0, len(bp2['boxes'])):
    plt.setp(bp2['boxes'][kk], facecolor='b')
ax.set_xscale('log')

fig.show()
