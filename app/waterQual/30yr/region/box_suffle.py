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

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

ep = 500
reTest = False

codeLst = ['00095', '00915', '00945', '00618']

corrLst = list()
rmseLst = list()
for iCode, code in enumerate(codeLst):
    siteNoLst = dictSite[code]
    corrMat = np.full([len(siteNoLst), 3], np.nan)
    rmseMat = np.full([len(siteNoLst), 3], np.nan)
    # comb and shuffle
    dataNameLst = ['rbWN5', 'rbWN5', 'rbWN5-S{}'.format(code)]
    trainSetLst = ['{}-B10'.format(code), 'comb-B10', 'comb-B10']
    trainCodeLst = [code, 'comb', 'comb']
    labelLst = ['QTFP_C', 'QFP_C', 'QFP_C']
    for k in range(3):
        dataName = dataNameLst[k]
        trainSet = trainSetLst[k]
        trainCode = trainCodeLst[k]
        label = labelLst[k]
        testSet = '{}-B10'.format(code)

        wqData = waterQuality.DataModelWQ(dataName)
        outName = '{}-{}-{}-{}'.format(dataName, trainCode, label, trainSet)
        master = basins.loadMaster(outName)
        yP, ycP = basins.testModel(
            outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
        p = ycP[:, master['varYC'].index(code)]
        ind = wqData.subset[testSet]
        ic = wqData.varC.index(code)
        o = wqData.c[ind, ic]
        info = wqData.info.iloc[ind].reset_index()
        for iS, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, k] = corr
            rmseMat[iS, k] = rmse
    corrLst.append(corrMat)
    rmseLst.append(rmseMat)


# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['solo', 'comb', 'shuffle']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrLst[k][:, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
