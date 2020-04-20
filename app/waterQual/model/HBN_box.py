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
outLst = ['HBN-Y8090-opt1', 'HBN-Y8090-opt2']
trainSet = 'Y8090'
testSet = 'Y0010'
errMatLst = list()
for outName in outLst:
    yp1, ycp1 = basins.testModel(outName, trainSet, wqData=wqData)
    yp2, ycp2 = basins.testModel(outName, testSet, wqData=wqData)
    errMat1 = wqData.errBySiteC(ycp1, wqData.varC, subset=trainSet)
    errMat2 = wqData.errBySiteC(ycp2, wqData.varC, subset=testSet)
    errMatLst.append(errMat1)
    errMatLst.append(errMat2)

codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
                '\n'+code for code in codeLst]
    labLst2 = ['train opt1','test opt1','train opt2', 'test opt2']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst:
            temp.append(errMat[:, ic, 1])
        dataBox.append(temp)
    title = 'correlation of {} group'.format(group)
    fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
    fig.suptitle(title)
    fig.show()
