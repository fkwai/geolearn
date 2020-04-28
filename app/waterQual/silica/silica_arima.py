import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear
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
outName = 'Silica64-Y8090-opt1'

wqData = waterQuality.DataModelWQ('Silica64')
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'
master = basins.loadMaster(outName)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)
ns = len(siteNoLst)
# calculate error from sequence
rmseMat = np.ndarray([ns, 3, 2])
corrMat = np.ndarray([ns, 3, 2])
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    rmseLSTM, corrLSTM = waterQuality.calErrSeq(dfPred[code], dfObs[code])
    dfP2 = wqLinear.loadSeq(siteNo, code, 'ARMA',
                            optT='Y8090', order=(5, 0, 0))
    rmseARMA, corrARMA = waterQuality.calErrSeq(dfP2[code], dfObs[code])
    dfP3 = wqLinear.loadSeq(siteNo, code, 'LR', optT='Y8090')
    rmseLR, corrLR = waterQuality.calErrSeq(dfP3[code], dfObs[code])
    rmseMat[k, :, :] = [rmseLSTM, rmseARMA, rmseLR]
    corrMat[k, :, :] = [corrLSTM, corrARMA, corrLR]


# box
for (errMat, title) in zip([rmseMat, corrMat], ['RMSE', 'Correlation']):
    dataBox = list()
    for k in range(2):
        temp = [errMat[:, i, k] for i in range(3)]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = ['LSTM', 'ARMA', 'LR']
    fig = figplot.boxPlot(dataBox, label1=label1, label2=label2, sharey=True)
    fig.suptitle(title)
    fig.show()
