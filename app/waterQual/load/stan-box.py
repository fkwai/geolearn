import importlib
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

wqData = waterQuality.DataModelWQ('stanRef', rmFlag=True)
dataName = os.path.join(kPath.dirWQ, 'trainData', 'stanRef.npz')
npz = np.load(dataName)
ul = npz['ul']
ut = npz['ut']
wqData.c = wqData.c*(ut-ul)+ul
outName = 'stanRef-Yodd-opt2'
trainSet = 'Yodd'
testSet = 'Yeven'
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
ind1 = wqData.subset[trainSet]
ind2 = wqData.subset[testSet]
ycP1 = ycP1.c*(ut[ind1, :]-ul[ind1, :])+ul[ind1, :]
ycP2 = ycP2.c*(ut[ind2, :]-ul[ind2, :])+ul[ind2, :]
errMatC1 = wqData.errBySiteC(
    ycP1, varC=master['varYC'], subset=trainSet,  rmExt=True)
errMatC2 = wqData.errBySiteC(
    ycP2, varC=master['varYC'], subset=testSet, rmExt=True)


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
importlib.reload(figplot)
saveDir = os.path.join(kPath.dirWQ, 'paper')
codePdf = usgs.codePdf
groupLst = [['00010', '00095', '00400', '80154', '70303', '00660',
             '00618', '00600', '00665', '00605', '71846', '00681'],
            ['00915', '00925', '00935', '00930', '00940', '00945',
             '00955', '00410', '00405', '00300', '00950', '00440']]
strLst = ['physical and nutrient variables', 'inorganics variables']
for k in range(2):
    codeLst = groupLst[k]
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['train', 'test']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in [errMatC1, errMatC2]:
            ind = np.where((countMat[:, ic, 0] > 20) &
                           (countMat[:, ic, 1] > 20))[0]
            temp.append(errMat[ind, ic, 1])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                          label2=labLst2, figsize=(12, 4), yRange=[0, 1])
    title = 'correlation of {}'.format(strLst[k])
    fig.suptitle(title)
    fig.show()
    fig.savefig(os.path.join(saveDir, 'box_group{}'.format(k)))
