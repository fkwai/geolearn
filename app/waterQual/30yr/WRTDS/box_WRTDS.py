
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
with open(os.path.join(dirSel, 'dictNB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 300
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst), 4], np.nan)

# LSTM
label = 'QT_C'
trainSet = 'comb-B10'
testSet = 'comb-A10'
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
            corrMat[iS, iCode, iT] = corr
            # rmseMat[iS, iCode, iT*2] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
file1 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'B10N5'))
dfCorr1 = pd.read_csv(file1, dtype={'siteNo': str}).set_index('siteNo')
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
for iCode, code in enumerate(codeLst):
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    corrMat[indS, iCode, 2] = dfCorr1.iloc[indS][code].values
    corrMat[indS, iCode, 3] = dfCorr2.iloc[indS][code].values


# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
# labLst2 = ['WRTDS train', 'WRTDS test', 'LSTM train', 'LSTM test']
labLst2 = ['WRTDS test', 'LSTM test']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    # for i in [2, 3, 0 ,1]:
    for i in [3, 1]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='br',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()

# p-values
testLst = ['p-value']
indLst = [[1, 3]]
codeStrLst = ['{} {}'.format(
    code, usgs.codePdf.loc[code]['shortName']) for code in codeLst]
dfS = pd.DataFrame(index=codeStrLst, columns=testLst)
for (test, ind) in zip(testLst, indLst):
    for k, code in enumerate(codeLst):
        data = [corrMat[:, k, x] for x in ind]
        [a, b], _ = utils.rmNan(data)
        # s, p = scipy.stats.ttest_ind(a, b, equal_var=False)
        s, p = scipy.stats.ttest_rel(a, b)
        dfS.loc[codeStrLst[k]][test] = p
pd.options.display.float_format = '{:,.2f}'.format
print(dfS)
