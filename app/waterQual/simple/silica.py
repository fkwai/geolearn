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

wqData = waterQuality.DataModelWQ('Silica16')
# test
codeLst = ['00955']
varPred = ['00060']+codeLst
dataName = 'Silica16'
outName = 'Silica16-Y8090-opt1'
trainset = 'Y8090'
testset = 'Y0010'

# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainset, varC=codeLst)
errMatQ1 = wqData.errBySiteQ(yP1, subset=trainset, varQ=['00060'])
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMatC2 = wqData.errBySiteC(ycP2, subset=testset, varC=codeLst)
errMatQ2 = wqData.errBySiteQ(yP2, subset=testset, varQ=['00060'])

# box
dataBox = list()
for k in range(2):
    temp = [errMatC1[:, 0, k], errMatC2[:, 0, k]]
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=['RMSE', 'Corr'], label2=[
                      'train', 'test'], sharey=False)
fig.show()

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# plot
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    nM = len(codeLst)+1
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    axplot.mapPoint(axM[0], lat, lon, errMatQ2[:, 0, 0], s=12)
    axM[0].set_title('streamflow')
    for k in range(1, nM):
        code = codeLst[k-1]
        # ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = '{} {}'.format(shortName, code)
        axplot.mapPoint(axM[k], lat, lon, errMatC2[:, 0, 0], s=12)
        axM[k].set_title(title)
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred['date'].values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    for k, var in enumerate(varPred):
        if var == '00060':
            styLst = '--'
            title = 'streamflow'
        else:
            styLst = '-*'
            shortName = codePdf.loc[var]['shortName']
            title = ' {} {}'.format(shortName, var)
        axplot.plotTS(axP[k], t, [dfPred[var], dfObs[var]], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k].set_title(title)


figplot.clickMap(funcMap, funcPoint)
