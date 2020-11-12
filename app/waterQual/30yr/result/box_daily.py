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
codeLst = sorted(usgs.newC)
ep = 300
reTest = True
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataLst = ['rbWN5', 'rbDN5']
label = 'QTFP_C'

corrMat = np.full([nSite, len(codeLst), len(dataLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(dataLst)], np.nan)
for k, dataName in enumerate(dataLst):
    wqData = waterQuality.DataModelWQ(dataName)
    trainSet = 'comb-B10'
    testSet = 'comb-A10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    master = basins.loadMaster(outName)
    yP, ycP = basins.testModel(
        outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
    for iCode, code in enumerate(codeLst):
        ic = wqData.varC.index(code)
        ind = wqData.subset[testSet]
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
            corrMat[iS, iCode, k] = corr
            rmseMat[iS, iCode, k] = rmse

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(len(dataLst)):
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='rb',
                      label2=['weekly,daily'], figsize=(12, 4), yRange=[0, 1])
fig.show()

