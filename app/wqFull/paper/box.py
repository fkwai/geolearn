
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
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=500)
codeLst = usgs.newC

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
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
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# box plot
dataPlot = list()
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codeLst]
for ic, code in enumerate(codeLst):
    dataPlot.append([corrL2[:, ic], corrW2[:, ic]])
    # dataPlot.append([corrL1[:, ic],corrL2[:, ic], corrW1[:, ic],corrW2[:, ic]])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label2=['LSTM', 'WRTDS'], label1=codeStrLst)
fig.show()
dirPaper = r'C:\Users\geofk\work\waterQuality\paper'
plt.savefig(os.path.join(dirPaper, 'box_all'))


#
a = np.nanmedian(corrW2, axis=0)
b = np.nanmean(corrW2**2 - corrL2**2, axis=0) 

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(b[k], a[k], usgs.codePdf.loc[codeLst[k]]['shortName'])
ax.plot(b, a, '*')
# ax.set_xlim([0.2, 1.2])
# ax.set_ylim([-1.5, 3])
plt.xscale('symlog')
fig.show()
