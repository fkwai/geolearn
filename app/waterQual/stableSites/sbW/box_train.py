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
wqData = waterQuality.DataModelWQ('sbW')
siteNoLst = wqData.info.siteNo.unique()
nSite = len(siteNoLst)

# single
labelLst = ['plain', 'ntnq']
cLst = 'bgmr'
labLst2 = ['plain train', 'plain test', 'ntnq train', 'ntnq test']

corrMat = np.full([nSite, len(codeLst), len(labelLst)*2], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(labelLst)*2], np.nan)
for iLab, label in enumerate(labelLst):
    for iCode, code in enumerate(codeLst):
        trainSet = '{}-Y1'.format(code)
        testSet = '{}-Y2'.format(code)
        outName = '{}-{}-{}-{}'.format('sbW', code, label, trainSet)
        master = basins.loadMaster(outName)
        ic = wqData.varC.index(code)
        for iT, subset in enumerate([trainSet, testSet]):
            yP, ycP = basins.testModel(
                outName, subset, wqData=wqData, ep=ep, reTest=reTest)
            ind = wqData.subset[subset]
            info = wqData.info.iloc[ind].reset_index()
            p = ycP[:, master['varYC'].index(code)]
            o = wqData.c[ind, ic]
            for iS, siteNo in enumerate(siteNoLst):
                indS = info[info['siteNo'] == siteNo].index.values
                rmse, corr = utils.stat.calErr(p[indS], o[indS])
                corrMat[iS, iCode, iLab*2+iT] = corr
                rmseMat[iS, iCode, iLab*2+iT] = rmse

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(len(labelLst)):
        temp.append(corrMat[:, k, i*2])
        temp.append(corrMat[:, k, i*2+1])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst=cLst,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
