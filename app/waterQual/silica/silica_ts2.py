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
outName = 'Silica64-Y8090-00955-opt1'
master = basins.loadMaster(outName)
dataName = master['dataName']
wqData = waterQuality.DataModelWQ(dataName)
trainset = '00955-Y8090'
testset = '00955-Y0010'
if master['varY'] is not None:
    plotVar = ['00060', '00955']
else:
    plotVar = ['00955']

# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainset, varC=master['varYC'])
if master['varY'] is not None:
    errMatQ1 = wqData.errBySiteQ(yP1, subset=trainset, varQ=master['varY'])
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMatC2 = wqData.errBySiteC(ycP2, subset=testset, varC=master['varYC'])
if master['varY'] is not None:
    errMatQ2 = wqData.errBySiteQ(yP2, subset=testset, varQ=master['varY'])

# box
dataBox = list()
for k in range(2):
    for var in plotVar:
        if var == '00060':
            temp = [errMatQ1[:, 0, k], errMatQ2[:, 0, k]]
        else:
            ic = master['varYC'].index(var)
            temp = [errMatC1[:, ic, k], errMatC2[:, ic, k]]
            dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=['RMSE', 'Corr'], label2=[
                      'train', 'test'], sharey=False)
fig.show()

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    nM = len(plotVar)
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    axM = np.array([axM]) if nM == 1 else axM
    for k, var in enumerate(plotVar):
        if var == '00060':
            axplot.mapPoint(axM[k], lat, lon, errMatQ2[:, 0, 1], s=12)
            axM[k].set_title('streamflow')
        else:
            ic = master['varYC'].index(var)
            shortName = codePdf.loc[var]['shortName']
            title = '{} {}'.format(shortName, var)
            axplot.mapPoint(axM[k], lat, lon, errMatC2[:, ic, 1], s=12)
            axM[k].set_title(title)
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    axP = np.array([axP]) if nM == 1 else axP
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred.index.values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')

    info1 = wqData.subsetInfo(trainset)
    info2 = wqData.subsetInfo(testset)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    tp = np.concatenate([t1, t2])
    yp = np.concatenate([ycP1[ind1], ycP2[ind2]])

    for k, var in enumerate(plotVar):
        rmse, corr = waterQuality.calErrSeq(dfPred[var], dfObs[var])
        tStr = '{}, rmse [{:.2f} {:.2f}], corr [{:.2f} {:.2f}]'.format(
            siteNo, rmse[0], rmse[1], corr[0], corr[1])
        if var == '00060':
            styLst = '--'
            title = 'streamflow '+tStr
            axplot.plotTS(axP[k], t, [dfPred[var], dfObs[var]], tBar=tBar,
                          legLst=['LSTM', 'observation'], styLst=styLst, cLst='br')
            axP[k].set_title(title)
        else:
            styLst = '-*'
            shortName = codePdf.loc[var]['shortName']
            title = shortName + ' ' + tStr
            axplot.plotTS(axP[k], t, dfPred[var], tBar=tBar,
                          legLst=['LSTM-sequence'], styLst='-', cLst='b')
            axplot.plotTS(axP[k], tp, yp, legLst=[
                'LSTM-sample'], styLst='*', cLst='g')
            axplot.plotTS(axP[k], t, dfObs[var],
                          legLst=['observation'], styLst='*', cLst='r')
            axP[k].set_title(title)


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

for ax in figP.axes:
    ax.set_xlim(np.datetime64('2015-01-01'), np.datetime64('2020-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1990-01-01'), np.datetime64('1995-01-01'))
figP.canvas.draw()

for ax in figP.axes:
    ax.set_xlim(np.datetime64('1980-01-01'), np.datetime64('2020-01-01'))
figP.canvas.draw()
