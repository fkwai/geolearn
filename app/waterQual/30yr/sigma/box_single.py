
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
reTest = True
dataName = 'rbWN5'
code = '00400'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite[code]
nSite = len(siteNoLst)

# LSTM
labelLst = ['FP_QC', 'FP_C', 'QT_C', 'QTFP_C', 'QFP_C']
corrMat = np.full([nSite, len(labelLst), 2], np.nan)
trainSet = '{}-B10'.format(code)
testSet = '{}-A10'.format(code)
for iLab, label in enumerate(labelLst):
    outName = '{}-{}-{}-{}-sigma'.format(dataName, code, label, trainSet)
    master = basins.loadMaster(outName)
    for iT, subset in enumerate([trainSet, testSet]):
        yP, ycP, sP, scP = basins.testModel(
            outName, subset, wqData=wqData, ep=ep, reTest=reTest)
        ind = wqData.subset[subset]
        info = wqData.info.iloc[ind].reset_index()
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
            corrMat[iS, iLab, iT] = corr
            # rmseMat[iS, iCode, iT*2] = rmse

# # WRTDS
# dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10')
# # dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
# file1 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'B10N5'))
# dfCorr1 = pd.read_csv(file1, dtype={'siteNo': str}).set_index('siteNo')
# file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
# dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
# for iCode, code in enumerate(codeLst):
#     indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
#     corrMat[indS, iCode, 4] = dfCorr1.iloc[indS][code].values
#     corrMat[indS, iCode, 5] = dfCorr2.iloc[indS][code].values

# plot box
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    # for i in [2, 3, 0 ,1]:
    for i in range(len(labelLst)):
        temp.append(corrMat[:, i, 1])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, widths=0.5,  figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
