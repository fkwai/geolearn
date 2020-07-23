from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import os
import time

doLst = list()
# doLst.append('messData')
if 'messData' in doLst:
    code = '00955'
    # manually copied and pasted
    wqData = waterQuality.DataModelWQ('Silica64Mess')
    q = wqData.q
    f = wqData.f
    c = wqData.c
    g = wqData.g
    # mess up c data
    ic = wqData.varC.index(code)
    cM = c.copy()
    np.random.shuffle(cM)
    cM[:, ic] = c[:, ic]
    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, 'Silica64Mess')
    np.savez(saveName, q=q, f=f, c=cM, g=g)


# test
wqData = waterQuality.DataModelWQ('Silica64')

outLst = ['Silica64-Y8090-opt1', 'Silica64-Y8090-opt2',
          'Silica64Mess-Y8090-opt1', 'Silica64Mess-Y8090-opt2']
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
    ic = master['varYC'].index(code)
    errMatLst1.append(errMatC1[:, ic, :])
    errMatLst2.append(errMatC2[:, ic, :])


# box
for k in range(2):
    dataBox = list()
    for errMatLst in [errMatLst1, errMatLst2]:
        temp = [errMat[:, k] for errMat in errMatLst]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = ['all C, Q in', 'all C, Q out','all C, Q in, messed', 'all C, Q out, messed']
    fig = figplot.boxPlot(dataBox, label1=label1, label2=label2, sharey=True)
    fig.suptitle('RMSE') if k == 0 else fig.suptitle('Correlation')
    fig.show()
