from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import json


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSiteN5 = json.load(f)
with open(os.path.join(dirSel, 'dictRB_Y30N2.json')) as f:
    dictSiteN2 = json.load(f)
codeLst = sorted(usgs.newC)

dictSite = dict()
for code in usgs.newC+['comb']:
    siteNoCode = list(set(dictSiteN2[code])-set(dictSiteN5['comb']))
    dictSite[code] = siteNoCode
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

corrMat = np.full([nSite, len(codeLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst)], np.nan)

ep = 500
reTest = True
wqData = waterQuality.DataModelWQ('rbWN2')
testSet = 'comb-B10'
label = 'FP_QC'
outName = '{}-{}-{}-{}-ungauge'.format('rbWN5', 'comb', label, testSet)
master = basins.loadMaster(outName)
yP, ycP = basins.testModel(
    outName, testSet, wqData=wqData, ep=ep, reTest=reTest)

dictP = dict()
dictO = dict()
for iCode, code in enumerate(codeLst):
    print(code)
    pLst = list()
    oLst = list()
    ic = wqData.varC.index(code)
    ind = wqData.subset[testSet]
    info = wqData.info.iloc[ind].reset_index()
    ic = wqData.varC.index(code)
    p = ycP[:, master['varYC'].index(code)]
    o = wqData.c[ind, ic]
    for siteNo in dictSite[code]:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iCode] = corr
        rmseMat[iS, iCode] = rmse
        pLst.append(np.nanmean(p[indS]))
        oLst.append(np.nanmean(o[indS]))
    dictP[code] = pLst
    dictO[code] = oLst


# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    temp.append(corrMat[:, k])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      figsize=(12, 4), yRange=[0, 1])
fig.show()

# 121 mean
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
    x = np.array(dictP[code])
    y = np.array(dictO[code])
    axplot.plot121(ax, x, y)
    rmse, corr = utils.stat.calErr(x, y)
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    axplot.titleInner(ax, titleStr)
    # print(i, j)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != 4:
        _ = ax.set_xticklabels([])
    # _ = ax.set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
fig.show()
