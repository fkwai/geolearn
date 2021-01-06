import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import matplotlib

dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
# ecoIdLst = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
#             'H', 'I', 'J', 'K', 'L', 'M', 'O', 'Q']
ecoIdLst = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'M', 'O', 'Q']

subsetLst = ['EcoB{}'.format(k) for k in ecoIdLst]

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# global model
nashLst1 = list()
rmseLst1 = list()
corrLst1 = list()
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
for subset in subsetLst:
    indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[subset]]
    nash1 = utils.stat.calNash(yP[indT:, indS, 0], yO[indT:, indS, 0])
    rmse1 = utils.stat.calRmse(yP[indT:, indS, 0], yO[indT:, indS, 0])
    corr1 = utils.stat.calCorr(yP[indT:, indS, 0], yO[indT:, indS, 0])
    nashLst1.append(nash1)
    rmseLst1.append(rmse1)
    corrLst1.append(corr1)

# local model
nashLst2 = list()
rmseLst2 = list()
corrLst2 = list()
for subset in subsetLst:
    testSet = subset
    outName = '{}-{}-B10-gs'.format(dataName, subset)
    yP, ycP = basinFull.testModel(
        outName, DM=dm, batchSize=20, testSet=testSet, reTest=False)
    yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
    nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
    rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
    corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
    nashLst2.append(nash2)
    rmseLst2.append(rmse2)
    corrLst2.append(corr2)

# plot box
matLst = [[corrLst2, corrLst1],
          [nashLst2, nashLst1]]
nameLst = ['corr', 'nash']
rangeLst = [[0.3, 1], [0, 1]]

matLst = [[rmseLst2, rmseLst1],
          [corrLst2, corrLst1],
          [nashLst2, nashLst1]]
nameLst = ['rmse', 'corr', 'nash']
rangeLst = [[0, 1], [0.3, 1], [0, 1]]
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'
for kk in range(3):
    name = nameLst[kk]
    mat = matLst[kk]
    yRange = rangeLst[kk]
    label1 = ecoIdLst
    label2 = ['Local', 'CONUS']
    dataBox = list()
    for k in range(len(subsetLst)):
        temp = list()
        temp.append(mat[0][k])
        temp.append(mat[1][k])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, widths=0.5, cLst='rb', label1=label1,
                          label2=None, figsize=(12, 4), yRange=yRange)
    saveFile = os.path.join(saveFolder, 'q_ref_ecoR_{}'.format(name))
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
    fig.show()
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='rb', label1=label1,
                      label2=['Local', 'CONUS'], legOnly=True)
saveFile = os.path.join(saveFolder, 'q_ecoR_legend')
fig.savefig(saveFile)
fig.savefig(saveFile+'.eps')
fig.show()

dfS = pd.DataFrame(index=ecoIdLst, columns=['rmse', 'corr', 'nash'])
for k, ecoId in enumerate(ecoIdLst):
    s, p = scipy.stats.ttest_ind(rmseLst2[k], rmseLst1[k])
    dfS.loc[ecoId]['rmse'] = p
    s, p = scipy.stats.ttest_ind(corrLst2[k], corrLst1[k])
    dfS.loc[ecoId]['corr'] = p
    s, p = scipy.stats.ttest_ind(nashLst2[k], nashLst1[k])
    dfS.loc[ecoId]['nash'] = p

rmse2 = np.concatenate(rmseLst2, axis=-1)
rmse1 = np.concatenate(rmseLst1, axis=-1)
s, p = scipy.stats.ttest_ind(rmse2, rmse1)
print(p)
corr2 = np.concatenate(corrLst2, axis=-1)
corr1 = np.concatenate(corrLst1, axis=-1)
s, p = scipy.stats.ttest_ind(corr2, corr1)
print(p)
nash2 = np.concatenate(nashLst2, axis=-1)
nash1 = np.concatenate(nashLst1, axis=-1)
s, p = scipy.stats.ttest_ind(nash2, nash1)
print(p)

# # significance test
# testLst = ['Q as target', 'Q as input']
# indLst = [[0, 2], [1, 2]]
# codeStrLst = ['{} {}'.format(
#     code, usgs.codePdf.loc[code]['shortName']) for code in codeLst]
# dfS = pd.DataFrame(index=codeStrLst, columns=testLst)
# for (test, ind) in zip(testLst, indLst):
#     for k, code in enumerate(codeLst):
#         data = [corrMat[:, k, x] for x in ind]
#         [a, b], _ = utils.rmNan(data)
#         s, p = scipy.stats.ttest_ind(a, b, equal_var=False)
#         # s, p = scipy.stats.ttest_rel(a, b)
#         dfS.loc[codeStrLst[k]][test] = p
# pd.options.display.float_format = '{:,.2f}'.format
# print(dfS)
