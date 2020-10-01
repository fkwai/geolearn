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
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)
codeLst = sorted(usgs.newC)
ep = 500
reTest = False
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)

# single
labelLst = ['Q_C', 'QF_C','QFP_C']
cLst = 'gbr'
labLst2 = ['Q', 'Q + Forcing', 'Q + Forcing chem']

corrMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    trainSet = 'comb-B16'
    testSet = 'comb-A16'
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
            corrMat[iS, iCode, iLab] = corr
            rmseMat[iS, iCode, iLab] = rmse

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(len(labelLst)):
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst=cLst,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()


# significance test
testLst = ['Q as target', 'Q as input']
indLst = [[0, 2], [1, 2]]
codeStrLst = ['{} {}'.format(
    code, usgs.codePdf.loc[code]['shortName']) for code in codeLst]
dfS = pd.DataFrame(index=codeStrLst, columns=testLst)
for (test, ind) in zip(testLst, indLst):
    for k, code in enumerate(codeLst):
        data = [corrMat[:, k, x] for x in ind]
        [a, b], _ = utils.rmNan(data)
        s, p = scipy.stats.ttest_ind(a, b, equal_var=False)
        # s, p = scipy.stats.ttest_rel(a, b)
        dfS.loc[codeStrLst[k]][test] = p
pd.options.display.float_format = '{:,.2f}'.format
print(dfS)
