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

dictRegion = {
    'PNV': [2, 3, 4, 5, 9, 11],
    'NUTR': [2, 3, 4, 5, 6, 7, 8, 9, 11, 14],
    'HLR': [3, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 20],
    'ECO': [5.3, 6.2, 8.1, 8.2, 8.3, 8.4, 9.2, 9.3, 9.4, 10.1, 11.1]
}
for region in list(dictRegion.keys()):
    for regionId in dictRegion[region]:
        if region == 'ECO':
            idLst = [int(x) for x in str(regionId).split('.')]
            regionId = '{:02d}{:02d}'.format(*idLst)
        else:
            regionId = '{:02d}'.format(regionId)

siteNoLst = wqData.siteNoLst
label = 'QFP_C'
ep = 500
reTest = False
codeLst = sorted(usgs.newC)
nSite = len(siteNoLst)
regionLst = list(dictRegion.keys())
corrMat = np.full([nSite, len(codeLst), len(regionLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(regionLst)], np.nan)

# region model
for k, region in enumerate(regionLst):
    for regionId in dictRegion[region]:
        if region == 'ECO':
            idLst = [int(x) for x in str(regionId).split('.')]
            regionId = '{:02d}{:02d}'.format(*idLst)
        else:
            regionId = '{:02d}'.format(regionId)
        trainSet = 'comb-{}{}-B10'.format(region, regionId)
        testSet = 'comb-{}{}-A10'.format(region, regionId)
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        master = basins.loadMaster(outName)
        yP, ycP = basins.testModel(
            outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
        ind = wqData.subset[testSet]
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
                corrMat[iS, iCode, k] = corr
                rmseMat[iS, iCode, k] = rmse

# global model
corrComb = np.full([nSite, len(codeLst)], np.nan)
rmseComb = np.full([nSite, len(codeLst)], np.nan)
trainSet = '{}-B10'.format('comb')
testSet = '{}-A10'.format('comb')
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
yP, ycP = basins.testModel(
    outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
ind = wqData.subset[testSet]
info = wqData.info.iloc[ind].reset_index()
siteNoTemp = info['siteNo'].unique()
for iCode, code in enumerate(codeLst):
    ic = wqData.varC.index(code)
    if len(wqData.c.shape) == 3:
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
    elif len(wqData.c.shape) == 2:
        p = ycP[:, master['varYC'].index(code)]
        o = wqData.c[ind, ic]
    for siteNo in siteNoTemp:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrComb[iS, iCode] = corr
        rmseComb[iS, iCode] = rmse


# plot box
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = regionLst+['global']
dataBox = list()
for k, code in enumerate(codeLst):
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    temp = list()
    for i in range(len(regionLst)):
        temp.append(corrMat[indS, k, i])
    temp.append(corrComb[indS, k])        
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
