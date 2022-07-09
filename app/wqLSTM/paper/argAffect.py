
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
DF = dbBasin.DataFrameBasin('N200')
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

code = '71846'
cVar = 'NITR_APP_KG_SQKM'
th = 5

indC = DF.varC.index(code)
x = dfG[cVar].values/1000
y = corrL2[:, indC]**2-corrW2[:, indC]**2

# attr vs diff
saveFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 6})

cVarStr = 'Estimated Nitrogen fertilizer [ton/sqkm]'
cVarStr2 = 'Nitrogen fertilizer'
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ic = codeLst.index(code)
sc = axplot.scatter121(axes[0], corrL2[:, indC],
                       corrW2[:, indC], x, size=15)
axes[0].set_xlabel(r'LSTM $\rho$')
axes[0].set_ylabel(r'WRTDS $\rho$')
fig.colorbar(sc, ax=axes[0])
axes[1].plot(x, y, '*')
axes[1].plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
axes[1].set_ylim([-0.6, 0.6])
axes[1].set_xlabel(cVarStr)
axes[1].set_ylabel('Rsq LSTM - Rsq WRTDS')
axes[1].vlines(th, -0.6, 0.6, color='r')
fig.suptitle('affect of {} on {}'.format(
    cVarStr, usgs.codePdf.loc[code]['shortName']))
fig.show()
fig.savefig(os.path.join(saveFolder, 'NHx1'))
fig.savefig(os.path.join(saveFolder, 'NHx1.svg'))

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
label1 = ['{}<={}\np-value={:.0e}'.format(cVarStr2, th, pLst[0]),
          '{}>{}\np-value={:.0e}'.format(cVarStr2, th, pLst[1])]
fig, axes = figplot.boxPlot(
    dataBox, label1=label1, label2=['LSTM', 'WRTDS'],
    widths=0.5, figsize=(6, 5), yRange=[0, 1])
fig.suptitle('affect of {} on {}'.format(
    cVarStr, usgs.codePdf.loc[code]['shortName']))
fig.show()
fig.savefig(os.path.join(saveFolder, 'NHx2'))
fig.savefig(os.path.join(saveFolder, 'NHx2.svg'))

ind = np.where((x > 2000) & (y < -0.1))[0]
fig, ax = plt.subplots(1, 1)
ax.plot(DF.t, DF.c[:, ind[0], indC], '*')
fig.show()
