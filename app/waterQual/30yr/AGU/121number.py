
import time
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
ep = 500
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst)], np.nan)

# LSTM
label = 'QFP_C'
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
        corrMat[iS, iCode] = corr
        # rmseMat[iS, iCode, iT*2] = rmse

# extract counts
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLstAll = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
countMatAll = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))
indCode = np.array([codeLstAll.index(code) for code in codeLst])
indSite = np.array([siteNoLstAll.index(siteNo) for siteNo in siteNoLst])
countMat = countMatAll[indSite, :, :]
countMat = countMat[:, :, indCode]
countMat1 = np.sum(countMat[:, :30, :], axis=1)
countMat2 = np.sum(countMat[:, 30:, :], axis=1)

# plot 121
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
fig, axes = plt.subplots(5, 4)
yticks = [-0.5, 0, 0.5, 1]
xticks = [0, 500, 1000]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    ind = codeLst.index(code)
    y = corrMat[:, ind]
    x = countMat1[:, ind]
    # c = np.argsort(countMat2[:, ind])
    ax.plot(x, y, '*')
    rmse, corr = utils.stat.calErr(x, y, rmExt=False)
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    axplot.titleInner(ax, titleStr)
    _ = ax.set_xlim([xticks[0], xticks[-1]])
    _ = ax.set_ylim([yticks[0], yticks[-1]])
    _ = ax.set_xticks(xticks[1:])
    _ = ax.set_yticks(yticks[1:])
    # print(i, j)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != 4:
        _ = ax.set_xticklabels([])
    # _ = ax.set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
# fig.colorbar()
fig.show()
