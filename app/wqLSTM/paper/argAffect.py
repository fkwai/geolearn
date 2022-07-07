
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
codeLst = usgs.varC


# LSTM
DF = dbBasin.DataFrameBasin('G200')


ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'

# trainSet = 'B10'
# testSet = 'A10'
label = 'QFPRT2C'
# label = 'FPRT2QC'
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
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan


dfG = gageII.readData(siteNoLst=DF.siteNoLst)


code = '00660'
cVar = 'PHOS_APP_KG_SQKM'
th = 1200

# code='00600'
# cVar='NITR_APP_KG_SQKM'

indC = DF.varC.index(code)
x = dfG[cVar].values
y = corrL2[:, indC]**2-corrW2[:, indC]**2

# attr vs diff
cMatLog = np.log(x+1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ic = codeLst.index(code)
sc = axplot.scatter121(axes[0], corrL2[:, indC],
                       corrW2[:, indC], x, size=10)
axes[0].set_xlabel('Corr LSTM')
axes[0].set_ylabel('Corr WRTDS')
fig.colorbar(sc, ax=axes[0])
axes[1].plot(x, y, '*')
axes[1].plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
axes[1].set_ylim([-0.5, 0.5])
axes[1].set_xlabel(cVar)
axes[1].set_ylabel('Rsq LSTM - Rsq WRTDS')
fig.suptitle('affect of {} on {} {}'.format(
    cVar, code, usgs.codePdf.loc[code]['shortName']))
fig.show()

# threshold
ind1 = np.where(x <= th)
ind2 = np.where(x > th)
dataBox = list()
pLst = list()
for ind in [ind1, ind2]:
    a = corrL2[ind, indC].flatten()
    b = corrW2[ind, indC].flatten()
    aa, bb = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.wilcoxon(aa, bb)
    dataBox.append([a, b])
    pLst.append(p)
label1 = ['<={:.3f}\np-value={:.0e}'.format(th, pLst[0]),
          '>{:.3f}\np-value={:.0e}'.format(th, pLst[1])]
fig, axes = figplot.boxPlot(
    dataBox, label1=label1, label2=['LSTM', 'WRTDS'],
    widths=0.5, figsize=(6, 4), yRange=[0, 1])
fig.suptitle('affect of {} on {} {}'.format(
    cVar, code, usgs.codePdf.loc[code]['shortName']))
fig.show()
