from hydroDL.app import DGSA
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sklearn.tree
import matplotlib.gridspec as gridspec
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin


DF = dbBasin.DataFrameBasin('G200')
codeLst = DF.varC
siteNoLst = DF.siteNoLst

# LSTM
ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
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

dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)


code = '00660'
ic = codeLst.index(code)
matY = corrL2[:, ic]**2-corrW2[:, ic]**2
pMat = dfG.values
dfS = DGSA.DGSA_light(
    pMat, matY, ParametersNames=dfG.columns.tolist(), n_clsters=5)
# ax = dfS.sort_values(by=0).plot.barh()
# plt.show()

dfSP = dfS.sort_values(by=0)
fig, ax = plt.subplots(1, 1)
x = range(len(dfSP))
cLst = list()
for b in (dfSP[0] > 1).tolist():
    cLst.append('r') if b is True else cLst.append('b')
ax.barh(x, dfSP[0].values, color=cLst)
ax.set_yticks(x)
ax.set_yticklabels(dfSP.index.tolist())
plt.tight_layout()
fig.show()
