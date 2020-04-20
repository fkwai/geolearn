import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

wqData = waterQuality.DataModelWQ('HBN')

# test
# codeLst = ['00955']
# trainset = '00955-Y8090'
# testset = '00955-Y0010'
# outName = 'HBN-00955-Y8090-opt2'

codeLst = ['00618']
trainset = '00618-Y8090'
testset = '00618-Y0010'
outName = 'HBN-00618-Y8090-opt2'

varPred = ['00060']+codeLst
dataName = 'HBN'
infoTrain = wqData.info.iloc[wqData.subset[trainset]].reset_index()
infoTest = wqData.info.iloc[wqData.subset[testset]].reset_index()

# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMat1 = wqData.errBySiteC(ycP1, codeLst, subset=trainset)
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMat2 = wqData.errBySiteC(ycP2, codeLst, subset=testset)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# linear reg data
varTup = (wqData.varF, wqData.varG, ['00060'], codeLst)
dataTup, statTup = wqData.transIn(subset=trainset, varTup=varTup)
dataTup2 = wqData.transIn(subset=testset, varTup=varTup, statTup=statTup)
varX = varTup[0]
statX = statTup[0]
mtdX = wqData.extractVarMtd(varX)
varY = varTup[2]
statY = statTup[2]
varYC = varTup[3]
statYC = statTup[3]
mtdYC = wqData.extractVarMtd(varYC)
xL1 = dataTup[0][-1, :, :]
yL1 = dataTup[2][-1, :, :]
ycL1 = dataTup[3]
xL2 = dataTup2[0][-1, :, :]
yL2 = dataTup2[2][-1, :, :]
ycL2 = dataTup2[3]

# point test l2
ycpL1 = np.full([len(infoTrain), 1], np.nan)
ycpL2 = np.full([len(infoTest), 1], np.nan)
for siteNo in siteNoLst:
    ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
    ind2 = infoTest[infoTest['siteNo'] == siteNo].index
    [xtL1, yctL1], _ = utils.rmNan([xL1[ind1, :], ycL1[ind1, :]])
    modelYC = LinearRegression().fit(xtL1, yctL1)
    yctL1 = wqData.transOut(modelYC.predict(xtL1), statYC, varYC)
    ycpL1[ind1] = yctL1
    if len(ind2) > 0:
        xtL2 = xL2[ind2, :]
        yctL2 = wqData.transOut(modelYC.predict(xtL2), statYC, varYC)
        ycpL2[ind2] = yctL2
errMatL1 = wqData.errBySiteC(ycpL1, codeLst, subset=trainset)
errMatL2 = wqData.errBySiteC(ycpL2, codeLst, subset=testset)

# box
dataBox = list()
errMatL2[:, 0, 0]
for k in range(2):
    temp = [errMat1[:, 0, k], errMatL1[:, 0, k],
            errMat2[:, 0, k], errMatL2[:, 0, k]]
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=['RMSE', 'Corr'], label2=[
                      'train LSTM', 'train LR', 'test LSTM', 'test LR'], sharey=False)
fig.show()

# plot
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    # shortName = codePdf.loc[code]['shortName']
    # title = 'RMSE of {} {}'.format(shortName, code)
    axplot.mapPoint(axM[0], lat, lon, errMat1[:, 0, 0], s=12)
    axplot.mapPoint(axM[1], lat, lon, errMatL2[:, 0, 0], s=12)
    figP, axP = plt.subplots(2, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred['date'].values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    # linear model
    ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
    [x1, y1, yc1], _ = utils.rmNan([xL1[ind1, :], yL1[ind1, :], ycL1[ind1, :]])
    modelY = LinearRegression().fit(x1, y1)
    modelYC = LinearRegression().fit(x1, yc1)
    sd = np.datetime64('1979-01-01')
    ed = np.datetime64('2020-01-01')
    dfX = waterQuality.readSiteX(siteNo, sd, ed, varX)
    x2 = transform.transInAll(dfX.values, mtdX, statLst=statX)
    y2 = modelY.predict(x2)
    yc2 = modelYC.predict(x2)
    yp = wqData.transOut(y2, statY, varY)
    ycp = wqData.transOut(yc2, statYC, varYC)
    code = codeLst[0]
    axplot.plotTS(axP[0], t, [dfPred['00060'], yp, dfObs['00060']], tBar=tBar,
                  legLst=['lstm', 'lr', 'obs'], styLst='---', cLst='bgr')
    axplot.plotTS(axP[1], t, [dfPred[code], ycp, dfObs[code]], tBar=tBar,
                  legLst=['lstm', 'lr', 'obs'], styLst='--*', cLst='bgr')


figplot.clickMap(funcMap, funcPoint)
