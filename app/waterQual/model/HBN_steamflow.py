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

dataName = 'HBN'
# outLst = ['HBN-00618-00955-all-Y8090-opt2', 'HBN-00618-00955-all-Y8090-opt4']
# testset = '00618-00955-all-Y0010'
outLst = ['HBN-Y8090-opt1', 'HBN-Y8090-opt4']
testset = 'Y0010'
siteNoLst = wqData.info['siteNo'].unique().tolist()

errMatLst = list()
for out in outLst:
    basins.testModelSeq(out, siteNoLst, wqData=wqData)
    yP2, ycP2 = basins.testModel(out, testset, wqData=wqData)
    errMat = wqData.errBySiteQ(yP2, ['00060'], subset=testset)
    errMatLst.append(errMat)

# # calculate error - adhoc
# siteNo = siteNoLst[0]

# tB = np.datetime64('2000-01-01')
# dfPred1, dfObs1 = basins.loadSeq(outLst[0], siteNo)
# a1 = dfPred1['00060']
# dfPred2, dfObs2 = basins.loadSeq(outLst[1], siteNo)
# b = dfPred2['00060']
# obs = dfObs1['00060']


a=errMatLst[0][:,0,1]
b=errMatLst[1][:,0,1]
plt.plot(a,b,'*')
plt.plot([0,1],[0,1],'-k')
plt.show()


# plot
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    axplot.mapPoint(axM[0], lat, lon, errMatLst[0][:, 0, 1], s=12)
    axplot.mapPoint(axM[1], lat, lon, errMatLst[1][:, 0, 1], s=12)
    figP, axP = plt.subplots(1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred1, dfObs1 = basins.loadSeq(outLst[0], siteNo)
    dfPred2, dfObs2 = basins.loadSeq(outLst[1], siteNo)
    t = dfPred1['date'].values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    axplot.plotTS(axP, t, [dfPred1['00060'], dfPred2['00060'], dfObs2['00060']], tBar=tBar,
                  legLst=['w/ C', 'w/o C', 'obs'], styLst='---', cLst='bgr')


figplot.clickMap(funcMap, funcPoint)
