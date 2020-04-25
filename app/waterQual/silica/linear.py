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


# input
outName = 'Silica64-Y8090-opt1'
testset = 'Y0010'
wqData = waterQuality.DataModelWQ('Silica64')

master = basins.loadMaster(outName)
dataName = master['dataName']
if wqData is None:
    wqData = waterQuality.DataModelWQ(dataName)
trainset = master['trainName']
infoTrain = wqData.info.iloc[wqData.subset[trainset]].reset_index()
infoTest = wqData.info.iloc[wqData.subset[testset]].reset_index()

# linear reg data
statTup = basins.loadStat(outName)
varTup = (master['varX'], master['varXC'], master['varY'], master['varYC'])
dataTup1 = wqData.transIn(subset=trainset, varTup=varTup, statTup=statTup)
dataTup2 = wqData.transIn(subset=testset, varTup=varTup, statTup=statTup)
dataTup1 = trainTS.dealNaN(dataTup1, master['optNaN'])
dataTup2 = trainTS.dealNaN(dataTup2, master['optNaN'])
varYC = varTup[3]
statYC = statTup[3]
x1 = dataTup1[0][-1, :, :]
yc1 = dataTup1[3]
x2 = dataTup2[0][-1, :, :]

# point test l2 - linear
nc = len(varYC)
matP1 = np.full([len(infoTrain), nc], np.nan)
matP2 = np.full([len(infoTest), nc], np.nan)
siteNoLst = infoTest['siteNo'].unique().tolist()
for siteNo in siteNoLst:
    ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
    ind2 = infoTest[infoTest['siteNo'] == siteNo].index
    xT1 = x1[ind1, :]
    ycT1 = yc1[ind1, :]
    for ic in range(nc):
        [xx, yy], iv = utils.rmNan([xT1, ycT1[:, ic]])
        if len(iv) > 0:
            modelYC = LinearRegression().fit(xx, yy)
            matP1[ind1, ic] = modelYC.predict(xT1)
            if len(ind2) > 0:
                xT2 = x2[ind2, :]
                matP1[ind2, ic] = modelYC.predict(xT2)
matO1 = wqData.transOut(matP1, statYC, varYC)
matO2 = wqData.transOut(matP2, statYC, varYC)

errMatL1 = wqData.errBySiteC(matO1, varYC, subset=trainset)
errMatL2 = wqData.errBySiteC(matO2, varYC, subset=testset)

# box
dataBox = list()
for k in range(nc):
    temp = [errMatL1[:, k, 1], errMatL2[:, k, 1]]
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox)
fig.show()

# auto regression
x1 = dataTup1[0]
yc1 = dataTup1[3]
x2 = dataTup2[0]

siteNo = siteNoLst[0]
ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
ind2 = infoTest[infoTest['siteNo'] == siteNo].index
xT1 = x1[:, ind1, :]
ycT1 = yc1[ind1, :]
xT2 = x1[:, ind2, :]
for ic in range(nc):
    [xx, yy], iv = utils.rmNan([xT1, ycT1[:, ic]])
    if len(iv) > 0:
        modelYC = LinearRegression().fit(xx, yy)
        matP1[ind1, ic] = modelYC.predict(xT1)
        if len(ind2) > 0:
            xT2 = x2[ind2, :]
            matP1[ind2, ic] = modelYC.predict(xT2)
