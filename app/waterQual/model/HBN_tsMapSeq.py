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

doLst = list()
doLst.append('subset')

codeLst = ['00618', '00955']
varPred = ['00060']+codeLst

dataName = 'HBN'
outName = 'HBN-opt1'
trainset = 'first80'
testset = 'last20'
# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMat1 = wqData.errBySite(ycP1, subset=trainset, varC=codeLst)
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMat2 = wqData.errBySite(ycP2, subset=testset, varC=codeLst)
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
    figM, axM = plt.subplots(2, 2, figsize=(8, 6))
    # shortName = codePdf.loc[code]['shortName']
    # title = 'RMSE of {} {}'.format(shortName, code)
    axplot.mapPoint(axM[0, 0], lat, lon, errMat1[:, 0, 0], s=12)
    axplot.mapPoint(axM[1, 0], lat, lon, errMat1[:, 1, 0], s=12)
    axplot.mapPoint(axM[0, 1], lat, lon, errMat2[:, 0, 0], s=12)
    axplot.mapPoint(axM[1, 1], lat, lon, errMat2[:, 1, 0], s=12)
    figP, axP = plt.subplots(3, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred['date'].values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    for k, var in enumerate(varPred):
        styLst = '--' if var == '00060' else '-*'
        axplot.plotTS(axP[k], t, [dfPred[var], dfObs[var]], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k].set_title(var)


figplot.clickMap(funcMap, funcPoint)
