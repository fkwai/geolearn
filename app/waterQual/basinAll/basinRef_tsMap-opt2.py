from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')


outName = 'basinRef-Y8090-opt2'
trainSet = 'Y8090'
testSet = 'Y0010'
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=300)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=300)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
q1, c1 = basins.getObs(outName, trainSet, wqData=wqData)
q2, c2 = basins.getObs(outName, testSet, wqData=wqData)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
# basins.testModelSeq(outName, siteNoLst, wqData=wqData, 
# ep=300)

basins.testModelSeq(outName, ['08070200'], wqData=wqData, ep=300)


# figure out number of sample
info1 = wqData.subsetInfo(trainSet)
info2 = wqData.subsetInfo(testSet)
dataTrain = wqData.extractSubset(trainSet)
dataTest = wqData.extractSubset(testSet)
ycT1 = dataTrain[3]
ycT2 = dataTest[3]
nc = ycT1.shape[1]
countMat = np.full([len(siteNoLst), nc, 2], 0)
for i, siteNo in enumerate(siteNoLst):
    indS1 = info1[info1['siteNo'] == siteNo].index.values
    indS2 = info2[info2['siteNo'] == siteNo].index.values
    for iC in range(nc):
        countMat[i, iC, 0] = np.count_nonzero(~np.isnan(ycT1[indS1, iC]))
        countMat[i, iC, 1] = np.count_nonzero(~np.isnan(ycT2[indS2, iC]))


# plot
codeSel = ['00300', '00915']
# codeSel = ['00600', '00605', '00405']
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf

indLst = list()
for k, code in enumerate(codeSel):
    ic = wqData.varC.index(code)
    ind = np.where((countMat[:, ic, 0] > 50) &
                   (countMat[:, ic, 1] > 50))[0]
    indLst.append(ind)
indAll = np.unique(np.concatenate(indLst))
siteNoLstP = [siteNoLst[i] for i in indAll]


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = 'correlation of {} {}'.format(shortName, code)
        ind = indLst[k]
        axplot.mapPoint(axM[k], lat[ind], lon[ind], errMatC2[ind, ic, 1], s=12)
        axM[k].set_title(title)
    figP, axP = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon[indAll], lat[indAll]


def funcPoint(iP, axP):
    print(iP)
    siteNo = siteNoLstP[iP]
    tBar = np.datetime64('2000-01-01')
    dfPred, dfObs = basins.loadSeq(outName, siteNo, ep=300)
    dfPred = dfPred[dfPred.index >= np.datetime64('1980-01-01')]
    dfObs = dfObs[dfObs.index >= np.datetime64('1980-01-01')]
    t = dfPred.index.values.astype(np.datetime64)
    for k, var in enumerate(codeSel):
        shortName = codePdf.loc[var]['shortName']
        title = '{} {} {}'.format(siteNo, shortName, var)
        styLst = ['-', '*']
        axplot.plotTS(axP[k], t, [dfPred[var].values, dfObs[var].values], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k].set_title(title)


plt.tight_layout
figplot.clickMap(funcMap, funcPoint)
