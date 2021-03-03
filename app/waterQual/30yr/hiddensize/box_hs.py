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

dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.siteNoLst
label = 'QFP_C'
ep = 500
reTest = False
codeLst = sorted(usgs.newC)
nSite = len(siteNoLst)

hsLst = [16, 32, 64, 128, 256, 512]
corrMat = np.full([nSite, len(codeLst), len(hsLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(hsLst)], np.nan)

for k, hs in enumerate(hsLst):
    code = 'comb'
    trainSet = '{}-B10'.format('comb')
    testSet = '{}-A10'.format('comb')
    outName = '{}-{}-{}-{}-hs{}'.format(dataName, code, label, trainSet, hs)
    master = basins.loadMaster(outName)
    yP, ycP = basins.testModel(
        outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
    ind = wqData.subset[testSet]
    info = wqData.info.iloc[ind].reset_index()
    siteNoTemp = info['siteNo'].unique()
    for iCode, code in enumerate(codeLst):
        ic = wqData.varC.index(code)
        if len(wqData.c.shape) == 3:
            p = yP[-1, :, master['varY'].index(code)]
            o = wqData.c[-1, ind, ic]
        elif len(wqData.c.shape) == 2:
            p = ycP[:, master['varYC'].index(code)]
            o = wqData.c[ind, ic]
        for siteNo in siteNoTemp:
            iS = siteNoLst.index(siteNo)
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, k] = corr
            rmseMat[iS, iCode, k] = rmse


# plot box
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = hsLst
dataBox = list()
for k, code in enumerate(codeLst):
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    temp = list()
    for i in range(len(hsLst)):
        temp.append(corrMat[indS, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
