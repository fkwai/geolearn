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


labelLst = ['FT2QC', 'QFT2C', 'QT2C']
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
ep = 500

# load global model
epG = 400  # models are killed before 500
DFA = dbBasin.DataFrameBasin('rmTK-B200')
yGLst1 = list()
yGLst2 = list()
# for label in labelLst:

label='QFT2C'
outName = '{}-{}-{}'.format('rmTK-B200', label, trainSet)
dictMaster = basinFull.loadMaster(outName)
yP1, ycP1 = basinFull.testModel(outName, DF=DFA, testSet=trainSet, ep=epG)
yP2, ycP2 = basinFull.testModel(outName, DF=DFA, testSet=testSet, ep=epG)
if dictMaster['varY'][0] == 'streamflow':
    yP1 = yP1[:, :, 1:]
    yP2 = yP2[:, :, 1:]
yGLst1.append(yP1)
yGLst2.append(yP2)

# load WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('rmTK-B200', trainSet, 'all.npz')
yW = np.load(os.path.join(dirWRTDS, fileName))['yW']
yW1 = DFA.extractSubset(yW, trainSet)
yW2 = DFA.extractSubset(yW, testSet)

# count
matB = (~np.isnan(DFA.c)).astype(int).astype(float)
matB1 = DFA.extractSubset(matB, trainSet)
matB2 = DFA.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)

# calculate stats
matObs = DFA.c
obs1 = DFA.extractSubset(matObs, trainSet)
obs2 = DFA.extractSubset(matObs, testSet)
strFunc='calCorr'
errFunc=getattr(utils.stat,strFunc)
corrL2=errFunc(yP2,obs2)**2
corrW2=errFunc(yW2,obs2)**2


matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL2, corrW2]
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

#
a = np.nanmean(matLR, axis=0)
b = np.nanmean(corrLst2[0]**2 - corrW2**2, axis=0)
c = np.nanmean(corrLst2[1]**2 - corrW2**2, axis=0)

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(a[k], (b[k]+c[k])/2, usgs.codePdf.loc[codeLst[k]]['shortName'])
    ax.plot([a, a], [b, c], c='0.5')
ax.plot(a, b, 'b*')
ax.plot(a, c, 'r*')
# ax.set_xlim([0.2, 1.2])
# ax.set_ylim([-1.5, 3])
# plt.xscale('symlog')
ax.axhline(0, color='k')
ax.axvline(0.33, color='k')

fig.show()

##
a = np.nanmean(matLR, axis=0)
b = np.nanmean(corrLst2[0]**2, axis=0)
c = np.nanmean(corrLst2[1]**2, axis=0)

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(a[k], (b[k]+c[k])/2, usgs.codePdf.loc[codeLst[k]]['shortName'])
    ax.plot([a, a], [b, c], c='0.5')
ax.plot(a, b, 'b*')
ax.plot(a, c, 'r*')
# ax.set_xlim([0.2, 1.2])
# ax.set_ylim([-1.5, 3])
# plt.xscale('symlog')

fig.show()
