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
codeLst = ['00955', '00915', '00405']
outLst = ['HBN-first50-{}'.format(x) for x in codeLst]
trainSet = 'first50'
testSet = 'last50'

outName = 'HBN-first50-opt1'
p1, o1 = basins.testModel(outName, trainSet)
p2, o2 = basins.testModel(outName, testSet)
errMat1 = wqData.errBySite(p1, subset=trainSet)
errMat2 = wqData.errBySite(p2, subset=testSet)
dataBox1 = list()
dataBox2 = list()
for code in codeLst:
    outName = 'HBN-first50-{}'.format(code)
    p1, o1 = basins.testModel(outName, trainSet)
    p2, o2 = basins.testModel(outName, testSet)
    varC = [code]
    err1 = wqData.errBySite(p1, subset=trainSet, varC=varC)
    err2 = wqData.errBySite(p2, subset=testSet, varC=varC)
    temp = list()
    ic = wqData.varC.index(code)
    temp.append(errMat2[:, ic, 1])
    temp.append(err2[:, 0, 1])
    dataBox1.append([errMat1[:, ic, 0],err1[:, 0, 0]])
    dataBox2.append([errMat2[:, ic, 0],err2[:, 0, 0]])

labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['all targets', 'single targets']
fig = figplot.boxPlot(dataBox1, label1=labLst1, label2=labLst2)
fig.suptitle('training correlation')
fig.show()

labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['all targets', 'single targets']
fig = figplot.boxPlot(dataBox2, label1=labLst1, label2=labLst2)
fig.suptitle('testing correlation')
fig.show()

