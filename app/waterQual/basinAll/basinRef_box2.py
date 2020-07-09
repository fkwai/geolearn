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

wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)

outLst = ['basinRef-Yeven-opt1', 'basinRef-Yeven-opt2']
trainSet = 'Yeven'
testSet = 'Yodd'

# outLst = ['basinRef-Yodd-opt1', 'basinRef-Yodd-opt2']
# trainSet = 'Yodd'
# testSet = 'Yeven'

ep = 500
errMatLst1, errMatLst2 = [list() for x in range(2)]

for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=ep)
    yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=ep)
    errMatC1 = wqData.errBySiteC(
        ycP1, varC=master['varYC'], subset=trainSet,  rmExt=True)
    errMatC2 = wqData.errBySiteC(
        ycP2, varC=master['varYC'], subset=testSet, rmExt=True)
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)


# figure out number of sample
siteNoLst = wqData.info['siteNo'].unique().tolist()
info1 = wqData.subsetInfo(trainSet)
info2 = wqData.subsetInfo(testSet)
dataTrain = wqData.extractSubset(trainSet)
dataTest = wqData.extractSubset(testSet)
ycT1 = dataTrain[3]
ycT2 = dataTest[3]
nc = ycT1.shape[1]
countMat = np.full([len(siteNoLst), nc, 2], 0)
for i, siteNo in enumerate(siteNoLst):
    indS1 = info1[info1['siteNo'] == siteNo].index.values
    indS2 = info2[info2['siteNo'] == siteNo].index.values
    for iC in range(nc):
        countMat[i, iC, 0] = np.count_nonzero(~np.isnan(ycT1[indS1, iC]))
        countMat[i, iC, 1] = np.count_nonzero(~np.isnan(ycT2[indS2, iC]))


# plot box
codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['train opt1', 'train opt2', 'test opt1', 'test opt2']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst1+errMatLst2:
            ind = np.where((countMat[:, ic, 0] > 20) &
                           (countMat[:, ic, 1] > 20))[0]
            temp.append(errMat[ind, ic, 1])
        dataBox.append(temp)
    title = 'correlation of {} group'.format(group)
    fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
    fig.suptitle(title)
    fig.show()
