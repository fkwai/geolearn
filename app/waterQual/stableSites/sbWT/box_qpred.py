from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

codeLst = sorted(usgs.varC)

ep = 500
reTest = False
wqData = waterQuality.DataModelWQ('sbWTQ')
siteNoLst = wqData.info.siteNo.unique()
nSite = len(siteNoLst)

# single
corrMat = np.full([nSite, len(codeLst), 3], np.nan)
rmseMat = np.full([nSite, len(codeLst), 3], np.nan)
for iLab, label in enumerate(['ntn', 'qpred', 'ntnq']):
    for iCode, code in enumerate(codeLst):
        trainSet = '{}-Y1'.format(code)
        testSet = '{}-Y2'.format(code)
        if label == 'qpred':
            outName = '{}-{}-{}-{}'.format('sbWTQ', code, label, trainSet)
        else:
            outName = '{}-{}-{}-{}'.format('sbWT', code, label, trainSet)
        master = basins.loadMaster(outName)
        ic = wqData.varC.index(code)
        # for iT, subset in enumerate([trainSet, testSet]):
        subset = testSet
        yP, ycP = basins.testModel(
            outName, subset, wqData=wqData, ep=ep, reTest=reTest)
        ind = wqData.subset[subset]
        info = wqData.info.iloc[ind].reset_index()
        if label == 'ntn':
            p = yP[-1, :, 1]
        else:
            p = yP[-1, :, 0]
        o = wqData.c[-1, ind, ic]
        for iS, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iLab] = corr
            rmseMat[iS, iCode, iLab] = rmse

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['Q as target ', 'sim Q as input', 'obs Q as input']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='bgr',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
