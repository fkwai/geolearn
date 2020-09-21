
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
import scipy

codeLst = sorted(usgs.varC)
ep = 500
reTest = False
wqData = waterQuality.DataModelWQ('sbWT')
siteNoLst = wqData.info.siteNo.unique()
nSite = len(siteNoLst)
labelLst = ['qonly', 'ntnq']
corrMat = np.full([nSite, len(codeLst), 4], np.nan)

dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
dfCorr = pd.read_csv(os.path.join(
    dirWrtds, '{}-{}-corr'.format('Y1', 'Y2')), index_col=0)
corrMat[:, :, 0] = dfCorr[codeLst].values
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F')
dfCorr = pd.read_csv(os.path.join(
    dirWrtds, '{}-{}-corr'.format('Y1', 'Y2')), index_col=0)
corrMat[:, :, 1] = dfCorr[codeLst].values

# single
for iLab, label in enumerate(labelLst):
    for iCode, code in enumerate(codeLst):
        trainSet = '{}-Y1'.format(code)
        testSet = '{}-Y2'.format(code)
        outName = '{}-{}-{}-{}'.format('sbWT', code, label, trainSet)
        master = basins.loadMaster(outName)
        ic = wqData.varC.index(code)
        # for iT, subset in enumerate([trainSet, testSet]):
        subset = testSet
        yP, ycP = basins.testModel(
            outName, subset, wqData=wqData, ep=ep, reTest=reTest)
        ind = wqData.subset[subset]
        info = wqData.info.iloc[ind].reset_index()
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
        for iS, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iLab+2] = corr

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['WRTDS', 'LSTM', 'LSTM + Forcing']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 2, 3]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='bgr',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()

# # significance test
# testLst = ['WRTDS vs LSTM']
# indLst = [[0, 2]]
# columns = list()
# for test in testLst:
#     columns.append(test+' stat')
#     columns.append(test+' pvalue')
# dfS = pd.DataFrame(index=codeLst, columns=columns)
# for (test, ind) in zip(testLst, indLst):
#     for k, code in enumerate(codeLst):
#         data = [corrMat[:, k, x] for x in ind]
#         [a, b], _ = utils.rmNan(data)
#         s, p = scipy.stats.ttest_ind(a, b, equal_var=False)
#         dfS.loc[code][test+' stat'] = s
#         dfS.loc[code][test+' pvalue'] = p
# dfS

# significance test
testLst = ['WRTDS vs LSTM', 'LSTM vs LSTM w/F']
codeStrLst = ['{} {}'.format(
    code, usgs.codePdf.loc[code]['shortName']) for code in codeLst]
indLst = [[0, 2], [2, 3]]
dfS = pd.DataFrame(index=codeStrLst, columns=testLst)
for (test, ind) in zip(testLst, indLst):
    for k, code in enumerate(codeLst):
        data = [corrMat[:, k, x] for x in ind]
        [a, b], _ = utils.rmNan(data)
        s, p = scipy.stats.ttest_ind(a, b, equal_var=False)
        dfS.loc[codeStrLst[k]][test] = p
pd.options.display.float_format = '{:,.2f}'.format
print(dfS)
