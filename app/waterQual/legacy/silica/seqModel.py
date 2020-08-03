
from hydroDL import kPath
from hydroDL.app import waterQuality2, waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins2, basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

importlib.reload(waterQuality2)
doLst = list()

if 'data' in doLst:
    # wqData = waterQuality.DataModelWQ('Silica64')
    # siteNoLst = wqData.siteNoLst
    # if not waterQuality.exist('Silica64Seq'):
    #     wqData = waterQuality2.DataModelWQ.new('Silica64Seq', siteNoLst)
    # importlib.reload(waterQuality2)
    # wqData = waterQuality2.DataModelWQ('Silica64Seq')
    temp = waterQuality2.DataModelWQ('Silica64')
    siteNoLst = temp.siteNoLst
    # wqData = waterQuality2.DataModelWQ.new('Silica64Seq', siteNoLst)
    wqData = waterQuality2.DataModelWQ('Silica64Seq')

if 'subset' in doLst:
    # subset only have silica
    code = '00955'
    ic = wqData.varQ.index(code)
    indC = np.where(~np.isnan(wqData.q[-1, :, ic]))[0]
    wqData.saveSubset(code, indC)
    indYr1 = waterQuality2.indYr(wqData.info.iloc[indC], yrLst=[1979, 2000])[0]
    wqData.saveSubset('{}-Y8090'.format(code), indYr1)
    indYr2 = waterQuality2.indYr(wqData.info.iloc[indC], yrLst=[2000, 2020])[0]
    wqData.saveSubset('{}-Y0010'.format(code), indYr2)

if 'train' in doLst:
    saveName = 'Silica64Seq-Y8090'
    caseName = basins2.wrapMaster(dataName='Silica64Seq', trainName='00955-Y8090',
                                  batchSize=[None, 200], varY=wqData.varQ, varYC=None,
                                  outName=saveName)
    basins2.trainModelTS(saveName)

if 'test' in doLst:
    wqData = waterQuality2.DataModelWQ('Silica64Seq')
    outName = 'Silica64Seq-Y8090'
    siteNoLst = wqData.siteNoLst
    basins2.testModelSeq(outName, siteNoLst, wqData=wqData)

outLst = ['Silica64Seq-Y8090', 'Silica64-Y8090-00955-opt1']
code = '00955'
wqData = waterQuality.DataModelWQ('Silica64')
siteNoLst = wqData.siteNoLst
ns = len(siteNoLst)
rmseMat = np.ndarray([ns, 2, 2])
corrMat = np.ndarray([ns, 2, 2])
for k, siteNo in enumerate(siteNoLst):
    for i, out in enumerate(outLst):
        print(k, siteNo)
        dfP, dfO = basins.loadSeq(out, siteNo)
        rmse, corr = waterQuality.calErrSeq(dfP[code], dfO[code])
        rmseMat[k, i, :] = rmse
        corrMat[k, i, :] = corr


# box
for (errMat, title) in zip([rmseMat, corrMat], ['RMSE', 'Correlation']):
    dataBox = list()
    for k in range(2):
        temp = [errMat[:, i, k] for i in range(2)]
        dataBox.append(temp)
    label1 = ['B2000', 'A2000']
    label2 = ['seq', 'point']
    fig = figplot.boxPlot(dataBox, label1=label1, label2=label2, sharey=True)
    fig.suptitle(title)
    fig.show()
