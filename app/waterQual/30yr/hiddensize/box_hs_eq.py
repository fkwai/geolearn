import matplotlib
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

dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.siteNoLst
label = 'QFP_C'
ep = 500
reTest = False
codeLst = sorted(usgs.newC)
nSite = len(siteNoLst)

hsLst = [16, 32, 64, 128, 256, 512]
epLst = [100, 200, 300, 400, 500]
corrMat = np.full([nSite, len(codeLst), len(hsLst), len(epLst), 2], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(hsLst), len(epLst), 2], np.nan)

for iH, hs in enumerate(hsLst):
    code = 'comb'
    trainSet = '{}-B10'.format('comb')
    testSet = '{}-A10'.format('comb')
    outName = '{}-{}-{}-{}-hs{}'.format(dataName, code, label, trainSet, hs)
    master = basins.loadMaster(outName)
    for iEp, ep in enumerate(epLst):
        for iSet, subset in enumerate([trainSet, testSet]):
            # subset = testSet
            yP, ycP = basins.testModel(
                outName, subset, wqData=wqData, ep=ep, reTest=reTest)
            ind = wqData.subset[subset]
            info = wqData.info.iloc[ind].reset_index()
            siteNoTemp = info['siteNo'].unique()
            for iCode, code in enumerate(codeLst):
                ic = wqData.varC.index(code)
                p = ycP[:, master['varYC'].index(code)]
                o = wqData.c[ind, ic]
                for siteNo in siteNoTemp:
                    iS = siteNoLst.index(siteNo)
                    indS = info[info['siteNo'] == siteNo].index.values
                    rmse, corr = utils.stat.calErr(p[indS], o[indS])
                    corrMat[iS, iCode, iH, iEp, iSet] = corr
                    rmseMat[iS, iCode, iH, iEp, iSet] = rmse


# plot box
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = epLst
dataBox = list()
for ic, code in enumerate(codeLst):
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    temp = list()
    # for i in range(len(hsLst)):
    #     temp.append(corrMat[indS, ic, i, 1])
    for i in range(len(epLst)):
        temp.append(corrMat[indS, ic, 5, i, 1])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
