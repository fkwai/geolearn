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
ep = 500
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)

rmCode = ['00010', '00095', '00400']
rmName = 'rmTKH'
# rmCode = ['00010', '00095']
# rmName = 'rmTK'
# rmCode = ['00010']
# rmName = 'rmT'

# single
labelLst = ['QT_C', 'QTFP_C', 'QFP_C']
cLst = 'grmbc'
labLst2 = [x.replace('_', '->') for x in labelLst]
codeLst = sorted(list(set(usgs.newC)-set(rmCode)))
siteNoLst = dictSite[rmName]
nSite = len(siteNoLst)


corrMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    trainSet = '{}-B10'.format(rmName)
    testSet = '{}-A10'.format(rmName)
    outName = '{}-{}-{}-{}'.format(dataName, rmName, label, trainSet)
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
            corrMat[iS, iCode, iLab] = corr
            rmseMat[iS, iCode, iLab] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat',
                        'WRTDS-W', 'B10-{}'.format(rmName))
corrWRTDS = np.full([nSite, len(codeLst), 2], np.nan)
# dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
file1 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'B10N5'))
dfCorr1 = pd.read_csv(file1, dtype={'siteNo': str}).set_index('siteNo')
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
for iCode, code in enumerate(codeLst):
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    corrWRTDS[indS, iCode, 0] = dfCorr1.iloc[indS][code].values
    corrWRTDS[indS, iCode, 1] = dfCorr2.iloc[indS][code].values


# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(len(labelLst)):
        temp.append(corrMat[:, k, i])
    temp.append(corrWRTDS[:, k, 1])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst=cLst,
                      label2=labLst2+['WRTDS'], figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
