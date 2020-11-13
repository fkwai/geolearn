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
reTest = True
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataName = 'rbWN5-WRTDS'
wqData = waterQuality.DataModelWQ(dataName)

# single
labelLst = ['QFP_C', 'QTFP_C', 'QT_C']
cLst = 'grmbc'
labLst2 = [x.replace('_', '->') for x in labelLst]


corrMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    trainSet = 'comb-B10'
    # testSet = 'comb-A10'
    testSet = 'comb-B10'
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
        for siteNo in dictSite[code[:5]]:
            iS = siteNoLst.index(siteNo)
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iLab] = corr
            rmseMat[iS, iCode, iLab] = rmse

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst3=[lab+'-WRTDS' for lab in labLst2]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(len(labelLst)):
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst=cLst,
                      label2=labLst3, figsize=(12, 4), yRange=[-1, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()

