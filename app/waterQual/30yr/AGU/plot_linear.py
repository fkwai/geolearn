
import matplotlib
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
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
dictLSTMLst = list()
# LSTM
labelLst = ['QFP_C']
for label in labelLst:
    dictLSTM = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outName, siteNo)
        dictLSTM[siteNo] = df
    dictLSTMLst.append(dictLSTM)
# WRTDS
dictWRTDS = dict()
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'Linear-W', 'B20', 'output')
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    # df = utils.time.datePdf(df)
    dictWRTDS[siteNo] = df
# Observation
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(siteNo, varLst=codeLst, freq='W')
    dictObs[siteNo] = df

# calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
dictLSTM = dictLSTMLst[1]
dictLSTM2 = dictLSTMLst[0]
corrMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
rmseMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictLSTM[siteNo][code].iloc[ind2].values
        v2 = dictWRTDS[siteNo][code].iloc[ind2].values
        v3 = dictObs[siteNo][code].iloc[ind2].values
        v4 = dictLSTM2[siteNo][code].iloc[ind2].values
        [v1, v2, v3, v4], ind = utils.rmNan([v1, v2, v3, v4])
        rmse1, corr1 = utils.stat.calErr(v1, v2, rmExt=False)
        rmse2, corr2 = utils.stat.calErr(v1, v3, rmExt=False)
        rmse3, corr3 = utils.stat.calErr(v2, v3, rmExt=False)
        rmse4, corr4 = utils.stat.calErr(v4, v3, rmExt=False)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3
        corrMat[indS, ic, 3] = corr4

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['LSTM vs WRTDS', 'LSTM vs Obs', 'WRTDS vs Obs']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='grb',
                      label2=labLst2, figsize=(20, 5), yRange=[0, 1])
fig.show()


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
    x = corrMat[:, ind, 1]
    y = corrMat[:, ind, 2]
    c = corrMat[:, ind, 0]
    out = axplot.scatter121(ax, x, y, c)
    rmse, corr = utils.stat.calErr(x, y)
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    _ = ax.set_xlim([ticks[0], ticks[-1]])
    _ = ax.set_ylim([ticks[0], ticks[-1]])
    _ = ax.set_yticks(ticks[1:])
    _ = ax.set_xticks(ticks[1:])
    axplot.titleInner(ax, titleStr)
    # print(i, j)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != 4:
        _ = ax.set_xticklabels([])
    # _ = ax.set_aspect('equal')
# plt.subplots_adjust(wspace=0, hspace=0)
    # fig.colorbar(out, ax=ax)
fig.show()

fig, ax = plt.subplots(1, 1)
code = '00095'
ind = codeLst.index(code)
x = corrMat[:, ind, 1]
y = corrMat[:, ind, 2]
c = corrMat[:, ind, 0]
out = axplot.scatter121(ax, x, y, c)
fig.colorbar(out, ax=ax)
fig.show()

# 121 LSTM inputs
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
fig, axes = plt.subplots(5, 4)
yticks = [-0.5, 0, 0.5, 1]
xticks = [-0.5, 0, 0.5, 1]
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    ind = codeLst.index(code)
    y = corrMat[:, ind, 1]
    x = corrMat[:, ind, 3]
    # c = np.argsort(countMat2[:, ind])
    axplot.plot121(ax, x, y)
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
