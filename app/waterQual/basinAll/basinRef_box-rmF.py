from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt


outLst = ['basinRef-Y8090-opt1', 'basinRef-Y8090-rmF-opt1']
trainSet = 'Y8090'
testSet = 'Y0010'
errMatLst1 = list()
errMatLst2 = list()
wqData1 = waterQuality.DataModelWQ('basinRef')
wqData2 = waterQuality.DataModelWQ('basinRef', rmFlag=True)


for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet)
    yP2, ycP2 = basins.testModel(outName, testSet)
    for wqData in [wqData1, wqData2]:
        errMatC1 = wqData.errBySiteC(
            ycP1, subset=trainSet, varC=master['varYC'])
        errMatC2 = wqData.errBySiteC(
            ycP2, subset=testSet, varC=master['varYC'])
        errMatLst1.append(errMatC1)
        errMatLst2.append(errMatC2)


# figure out number of sample
siteNoLst = wqData1.info['siteNo'].unique().tolist()
nc = ycP1.shape[1]
countMat1 = np.full([len(siteNoLst), nc, 2], 0)
countMat2 = np.full([len(siteNoLst), nc, 2], 0)
for wqData, countMat in zip([wqData1, wqData2], [countMat1, countMat2]):
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    dataTrain = wqData.extractSubset(trainSet)
    dataTest = wqData.extractSubset(testSet)
    ycT1 = dataTrain[3]
    ycT2 = dataTest[3]
    for i, siteNo in enumerate(siteNoLst):
        indS1 = info1[info1['siteNo'] == siteNo].index.values
        indS2 = info2[info2['siteNo'] == siteNo].index.values
        for iC in range(nc):
            countMat[i, iC, 0] = np.count_nonzero(~np.isnan(ycT1[indS1, iC]))
            countMat[i, iC, 1] = np.count_nonzero(~np.isnan(ycT2[indS2, iC]))
countMatLst = [countMat1, countMat2, countMat1, countMat2]

# plot box
codePdf = usgs.codePdf
codeLst = ['00660', '00665', '00600', '00605', '00618', '71846', '00950']
# codeLst = codePdf[codePdf.group == group].index.tolist()
indLst = [wqData.varC.index(code) for code in codeLst]
labLst1 = [codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train all test all', 'train all test rmFlag',
           'train rmFlag test all', 'train rmFlag test rmFlag']
dataBox = list()
rho = 20
for ic in indLst:
    temp = list()
    for errMat, countMat in zip(errMatLst2, countMatLst):
        ind = np.where((countMat[:, ic, 0] > 20) &
                       (countMat[:, ic, 1] > 20))[0]
        temp.append(errMat[ind, ic, 1])
        # temp.append(errMat[:, ic, 1])
    dataBox.append(temp)
title = 'test correlation of sites with >{} samples'.format(rho)
fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2, figsize=(12, 6))
fig.suptitle(title)
fig.show()
