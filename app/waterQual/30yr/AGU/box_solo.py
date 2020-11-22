import matplotlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import json

codeLst = sorted(usgs.newC)
ep = 500
reTest = False

dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.info.siteNo.unique().tolist()
nSite = len(siteNoLst)
label = 'QTFP_C'
# single
corrMat = np.full([nSite, len(codeLst), 2], np.nan)
rmseMat = np.full([nSite, len(codeLst), 2], np.nan)
for iCode, code in enumerate(codeLst):
    trainSet = '{}-B10'.format(code)
    testSet = '{}-A10'.format(code)
    outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
    master = basins.loadMaster(outName)
    ic = wqData.varC.index(code)
    for iT, subset in enumerate([trainSet, testSet]):
        yP, ycP = basins.testModel(
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
        for iS, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iT] = corr
            rmseMat[iS, iCode, iT] = rmse


# comb
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
ep = 500
corrComb = np.full([nSite, len(codeLst), 2], np.nan)
rmseComb = np.full([nSite, len(codeLst), 2], np.nan)
trainSet = '{}-B10'.format('comb')
testSet = '{}-A10'.format('comb')
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
for iT, subset in enumerate([trainSet, testSet]):
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
            corrComb[iS, iCode, iT] = corr
            rmseComb[iS, iCode, iT] = rmse

matplotlib.rcParams.update({'font.size': 13})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# plot box1
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train', 'test']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='cm',
                      label2=labLst2, figsize=(20, 5), yRange=[0, 1])
fig.show()


# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train solo', 'test solo', 'train comb',  'test comb']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1]:
        temp.append(corrMat[:, k, i])
    for i in [0, 1]:
        temp.append(corrComb[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='cmbr',
                      label2=labLst2, figsize=(20, 5), yRange=[0, 1])
fig.show()
