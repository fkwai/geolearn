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

wqData = waterQuality.DataModelWQ('HBN')
figFolder = os.path.join(kPath.dirWQ, 'HBN')

# compare of opt1-4
outLst = ['HBN-first50-opt1', 'HBN-first50-opt2']
trainSet = 'first50'
testSet = 'last50'
errMatLst1, errMatLst2 = [list(), list()]
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)
    master = basins.loadMaster(outName)
    varAll = master['varYC']

codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    outGroupLst = ['HBN-first50-opt1-'+group, 'HBN-first50-opt2-'+group]
    errMatGroup1, errMatGroup2 = [list(), list()]
    for outName in outGroupLst:
        master = basins.loadMaster(outName)
        varGroup = master['varYC']
        p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
        p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
        errMat1 = wqData.errBySite(p1, varC=varGroup, subset=trainSet)
        errMat2 = wqData.errBySite(p2, varC=varGroup, subset=testSet)
        errMatGroup1.append(errMat1)
        errMatGroup2.append(errMat2)
    indLst = [wqData.varC.index(code) for code in varGroup]
    labLst1 = [codePdf.loc[code]['shortName']+'\n'+code for code in varGroup]
    labLst2 = ['opt1-all', 'opt2-all', 'opt1-group', 'opt2-group']
    for errMatLst, errMatGroup, train in zip([errMatLst1, errMatLst2],
                                            [errMatGroup1, errMatGroup2],
                                            ['train', 'test']):
        dataBox = list()
        for k, ic in enumerate(indLst):
            temp = list()
            for errMat in errMatLst:
                temp.append(errMat[:, ic, 1])
            for errMat in errMatGroup:
                temp.append(errMat[:, k, 1])
            dataBox.append(temp)
        fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
        figName = 'box_{}_{}_group'.format(train, group)
        title = '{} correlation of {} group'.format(train, group)
        fig.suptitle(title)
        fig.savefig(os.path.join(figFolder, figName))
