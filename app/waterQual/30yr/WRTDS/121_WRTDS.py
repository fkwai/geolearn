
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 300
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst), 2], np.nan)

# LSTM
label = 'QT_C'
trainSet = 'comb-B10'
testSet = 'comb-A10'
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
subset = testSet
yP, ycP = basins.testModel(
    outName, subset, wqData=wqData, ep=ep, reTest=reTest)
ind = wqData.subset[subset]
info = wqData.info.iloc[ind].reset_index()
for iCode, code in enumerate(codeLst):
    ic = wqData.varC.index(code)
    if len(wqData.c.shape) == 3:
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
    elif len(wqData.c.shape) == 2:
        p = ycP[:, master['varYC'].index(code)]
        o = wqData.c[ind, ic]
    for siteNo in dictSite[code]:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iCode, 0] = corr
        # rmseMat[iS, iCode, iT*2] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
for iCode, code in enumerate(codeLst):
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    corrMat[indS, iCode, 1] = dfCorr2.iloc[indS][code].values

# plot 121
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
fig, axes = plt.subplots(5, 4)
ticks = [-0.5, 0, 0.5, 1]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    ind = codeLst.index(code)
    axplot.plot121(ax, corrMat[:, ind, 0], corrMat[:, ind, 1])
    rmse, corr = utils.stat.calErr(corrMat[:, ind, 0], corrMat[:, ind, 1])
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    _ = ax.set_xlim([ticks[0], ticks[-1]])
    _ = ax.set_ylim([ticks[0], ticks[-1]])
    _ = ax.set_yticks(ticks)
    _ = ax.set_xticks(ticks)
    axplot.titleInner(ax, titleStr)
    # print(i, j)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != 4:
        _ = ax.set_xticklabels([])
    # _ = ax.set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
fig.show()

fig, axes = plt.subplots(1, 2)
for k, code in enumerate(['00010', '00300']):
    ax = axes[k]
    ind = codeLst.index(code)
    axplot.plot121(ax, corrMat[:, ind, 0], corrMat[:, ind, 1])
    rmse, corr = utils.stat.calErr(corrMat[:, ind, 0], corrMat[:, ind, 1])
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    ax.set_title(titleStr)
fig.show()
