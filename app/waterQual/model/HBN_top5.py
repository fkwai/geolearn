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

wqData = waterQuality.DataModelWQ('HBN')

# test
codeLst = ['00955']
varPred = ['00060']+codeLst
dataName = 'HBN'
outName = 'HBN-00955-Y8090-opt2'
trainset = '00955-Y8090'
testset = '00955-Y0010'
# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMat1 = wqData.errBySiteC(ycP1, codeLst, subset=trainset)
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMat2 = wqData.errBySiteC(ycP2, codeLst, subset=testset)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# find top 5 sites and plot
infoTrain = wqData.info.iloc[wqData.subset[trainset]]
tabCount = infoTrain.groupby('siteNo').count(
).sort_values('date', ascending=False)


tBar = np.datetime64('2000-01-01')
figP, axP = plt.subplots(5, 1, figsize=(8, 6))
for k in range(5):
    siteNo = tabCount.index[k+20]
    nS = tabCount['date'][k+20]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred['date'].values.astype(np.datetime64)
    axplot.plotTS(axP[k], t, [dfPred['00955'], dfObs['00955']], tBar=tBar,
                  legLst=['pred', 'obs'], styLst='-*', cLst='br')
    axP[k].set_title('{} #samples = {}'.format(siteNo, nS))
    figP.show()


# plot
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(8, 6))
    axplot.mapPoint(axM[0], lat, lon, errMat1[:, 0, 0], s=12)
    axplot.mapPoint(axM[1], lat, lon, errMat2[:, 0, 0], s=12)
    figP, axP = plt.subplots(2, 1, figsize=(8, 6))
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
