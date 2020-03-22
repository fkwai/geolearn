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
import matplotlib.pyplot as plt

# master = basins.loadMaster('HBN-opt2')
# wqData = waterQuality.DataModelWQ(master['dataName'])
# p1, o1 = basins.testModel('HBN-first50-opt2', 'first50', wqData=wqData)

# outLst = ['HBN-opt1', 'HBN-opt2', 'HBN-opt3', 'HBN-opt4']
# trainSet = 'first80'
# testSet = 'last20'
# outLst = ['HBN-first50-opt1', 'HBN-first50-opt2',
#           'HBN-first50-opt3', 'HBN-first50-opt4']
outLst = ['HBN-first50-opt1', 'HBN-first50-opt2']
trainSet = 'first50'
testSet = 'last50'

pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
master = basins.loadMaster('HBN-opt1')
wqData = waterQuality.DataModelWQ(master['dataName'])
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    pLst1.append(p1)
    pLst2.append(p2)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)


# box plot
codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst = [codePdf.loc[code]['shortName']+'\n'+code for code in codeLst]
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst2:
            temp.append(errMat[:, ic, 1])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, label1=labLst)
    fig.show()

 # plot
# get location
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codeSel = ['00955', '00940', '00915']
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 2, figsize=(8, 6))
    for k in range(len(codeSel)):
        ic = icLst[k]
        axplot.mapPoint(axM[k, 0], lat, lon, errMat2[:, ic, 0], s=6)
        axplot.mapPoint(axM[k, 1], lat, lon, errMat2[:, ic, 1], s=6)
    figP, axP = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.extractSubsetInfo(trainSet)
    info2 = wqData.extractSubsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    x1 = p1[ind1]
    y1 = o1[ind1]
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    x2 = p2[ind2]
    y2 = o2[ind2]
    icLst = [wqData.varC.index(code) for code in codeSel]
    for k, ic in enumerate(icLst):
        axplot.plotTS(axP[k], t1, [x1[:, ic], y1[:, ic]], cLst='rb')
        axplot.plotTS(axP[k], t2, [x2[:, ic], y2[:, ic]], cLst='yg')


figplot.clickMap(funcMap, funcPoint)
