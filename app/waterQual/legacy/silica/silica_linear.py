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
import pandas as pd
import matplotlib.pyplot as plt

# test
wqData = waterQuality.DataModelWQ('Silica64')

outLst = ['Silica64-Y8090-opt1', 'Silica64-Y8090-00955-opt1']
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'

errMatLst1 = list()
errMatLst2 = list()
for outName in outLst:
    master = basins.loadMaster(outName)
    dataName = master['dataName']
    # wqData = waterQuality.DataModelWQ(dataName)
    # point test
    yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
    errMatC1 = wqData.errBySiteC(ycP1, subset=trainset, varC=master['varYC'])
    yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
    errMatC2 = wqData.errBySiteC(ycP2, subset=testset, varC=master['varYC'])

    ycL1, ycL2 = basins.modelLinear(
        outName, testset, trainset=trainset, wqData=wqData)
    errMatL1 = wqData.errBySiteC(ycL1, subset=trainset, varC=master['varYC'])
    errMatL2 = wqData.errBySiteC(ycL2, subset=testset, varC=master['varYC'])

    ic = master['varYC'].index(code)
    errMatLst1.append(errMatC1[:, ic, :])
    errMatLst2.append(errMatC2[:, ic, :])

    errMatLst1.append(errMatL1[:, ic, :])
    errMatLst2.append(errMatL2[:, ic, :])


# box
for k in range(2):
    dataBox = list()
    for errMatLst in [errMatLst1, errMatLst2]:
        temp = [errMat[:, k] for errMat in errMatLst]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = ['all C', 'all C, linear','single C', 'single C, linear']
    fig = figplot.boxPlot(dataBox, label1=label1, label2=label2, sharey=True)
    fig.suptitle('RMSE') if k == 0 else fig.suptitle('Correlation')
    fig.show()
