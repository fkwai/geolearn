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
import pandas as pd
import scipy

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.info.siteNo.unique().tolist()
nSite = len(siteNoLst)
label = 'QFP_C'

corrMat = np.full([nSite, len(codeLst), 4], np.nan)
rmseMat = np.full([nSite, len(codeLst), 4], np.nan)

# comb
trainSet = '{}-B16'.format('comb')
testSet = '{}-A16'.format('comb')
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
            rmseMat[iS, iCode, iT] = rmse

# seq test
for iS, siteNo in enumerate(siteNoLst):
    dfP = basins.loadSeq(outName, siteNo)
    dfO = waterQuality.readSiteTS(siteNo, codeLst, freq=wqData.freq)
    yr = pd.DatetimeIndex(dfP.index).year
    for iC, code in enumerate(codeLst):
        if siteNo in dictSite[code]:
            o1 = dfO[code].values[(yr <= 2016) & (yr >= 1980)]
            p1 = dfP[code].values[(yr <= 2016) & (yr >= 1980)]
            o2 = dfO[code].values[yr > 2016]
            p2 = dfP[code].values[yr > 2016]
            rmse1, corr1 = utils.stat.calErr(p1, o1)
            rmse2, corr2 = utils.stat.calErr(p2, o2)
            corrMat[iS, iC, 2] = corr1
            corrMat[iS, iC, 3] = corr2
            rmseMat[iS, iC, 2] = rmse1
            rmseMat[iS, iC, 3] = rmse2

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train - 1 year forward', 'train - full sequence',
           'test - full sequence', 'test - full sequence']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in range(4):
        temp.append(corrMat[:, k, i])
        # temp.append(rmseMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()

# p-values
testLst = ['training', 'testing']
indLst = [[0, 2], [1, 3]]
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
