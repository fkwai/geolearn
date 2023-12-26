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
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# stat
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
matObs = DF.c
# obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
# yP1 = DF.extractSubset(yP, trainSet)
yP2 = DF.extractSubset(yP, testSet)
# yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)

# stat
statFunc=utils.stat.calCorr
statFunc=utils.stat.calNash


# corrL1 = statFunc(yP1, obs1)
corrL2 = statFunc(yP2, obs2)
# corrW1 = statFunc(yW1, obs1)
corrW2 = statFunc(yW2, obs2)
importlib.reload(utils.stat)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL2, corrW2]:
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
b = np.nanmean(corrL2**2 - corrW2**2, axis=0)
c = np.nanmean(corrL2**2 - corrW2**2, axis=0)
b = np.nanmean(corrL2 - corrW2, axis=0)
c = np.nanmean(corrL2 - corrW2, axis=0)

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(a[k], (b[k]+c[k])/2, usgs.codePdf.loc[codeLst[k]]['shortName'])
    ax.plot([a, a], [b, c], c='0.5')
ax.plot(a, b, 'b*')
ax.plot(a, c, 'r*')
ax.set_xlim([0.2, 1.2])
ax.set_ylim([-1, 1])
# plt.xscale('symlog')
# ax.axhline(0, color='k')
# ax.axvline(0.33, color='k')

fig.show()

# ##
# a = np.nanmean(matLR, axis=0)
# b = np.nanmean(corrLst2[0]**2, axis=0)
# c = np.nanmean(corrLst2[1]**2, axis=0)

# fig, ax = plt.subplots(1, 1)
# for k in range(len(codeLst)):
#     ax.text(a[k], (b[k]+c[k])/2, usgs.codePdf.loc[codeLst[k]]['shortName'])
#     ax.plot([a, a], [b, c], c='0.5')
# ax.plot(a, b, 'b*')
# ax.plot(a, c, 'r*')
# # ax.set_xlim([0.2, 1.2])
# # ax.set_ylim([-1.5, 3])
# # plt.xscale('symlog')

# fig.show()
