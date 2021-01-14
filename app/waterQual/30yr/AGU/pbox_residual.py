import matplotlib
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
    dictSite = json.load(f)
codeLst = [code+'-R' for code in sorted(usgs.newC)]
ep = 500
reTest = False
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataName = 'rbWN5-WRTDS'
wqData = waterQuality.DataModelWQ(dataName)

# single
label = 'QTFP_C'
corrMat = np.full([nSite, len(codeLst), 2], np.nan)
rmseMat = np.full([nSite, len(codeLst), 2], np.nan)

trainSet = 'comb-B10'
testSet = 'comb-A10'
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
for iT, subset in enumerate([trainSet, testSet]):
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData, ep=ep, reTest=reTest)
    for iCode, code in enumerate(codeLst):
        ic = wqData.varC.index(code)
        ind = wqData.subset[subset]
        info = wqData.info.iloc[ind].reset_index()
        if len(wqData.c.shape) == 3:
            p = yP[-1, :, master['varY'].index(code)]
            o = wqData.c[-1, ind, ic]
        elif len(wqData.c.shape) == 2:
            p = ycP[:, master['varYC'].index(code)]
            o = wqData.c[ind, ic]
        for siteNo in dictSite[code[:5]]:
            iS = siteNoLst.index(siteNo)
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iT] = corr
            rmseMat[iS, iCode, iT] = rmse

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst3 = ['train', 'test']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(2):
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='br',
                      label2=labLst3, figsize=(20, 5), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()

