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

wqData = waterQuality.DataModelWQ('basinRef')


outLst = ['basinRef-first50-opt1', 'basinRef-first50-opt2']
trainSet = 'first50'
testSet = 'last50'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
    yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
    errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
    errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
    pLst1.append(ycP1)
    pLst2.append(ycP2)
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)

# figure out number of sample
info = wqData.info
siteNoLst = info['siteNo'].unique().tolist()
ycT = wqData.c
nc = ycT.shape[1]
countMat = np.full([len(siteNoLst), nc], 0)
for i, siteNo in enumerate(siteNoLst):
    indS = info[info['siteNo'] == siteNo].index.values
    for iC in range(nc):
        countMat[i, iC] = np.count_nonzero(~np.isnan(ycT[indS, iC]))

# plot box
codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['train opt1', 'train opt2', 'test opt2', 'test opt2']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst1+errMatLst2:
            ind = np.where(countMat[:, ic] > 50)[0]
            temp.append(errMat[:, ic, 1])
        dataBox.append(temp)
    title = 'correlation of {} group'.format(group)
    fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
    fig.suptitle(title)
    fig.show()
