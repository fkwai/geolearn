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

wqData = waterQuality.DataModelWQ('basinRef')


outLst = ['basinRef-first50-opt1', 'basinRef-first50-opt2']
trainSet = 'first50'
testSet = 'last50'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
    yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
    errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
    errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
    pLst1.append(ycP1)
    pLst2.append(ycP2)
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)

q1, c1 = basins.getObs(outName, trainSet, wqData=wqData)
q2, c2 = basins.getObs(outName, testSet, wqData=wqData)

# plot
codeSel = ['00955', '00940', '00915']
# codeSel = ['00600', '00605', '00405']
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = 'correlation of {} {}'.format(shortName, code)
        axplot.mapPoint(axM[k], lat, lon, errMatC2[:, ic, 1], s=12)
        axM[k].set_title(title)
    figP, axP = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    t = np.concatenate([t1, t2])
    dataLst = list()
    styLst = list()
    cLst = 'rbgcmy'
    for p1, p2, c in zip(pLst1, pLst2, cLst[:len(pLst2)]):
        x = np.concatenate([p1[ind1], p2[ind2]])
        dataLst.append(x)
        styLst.append('--'+c)
    y = np.concatenate([c1[ind1], c2[ind2]])
    dataLst.append(y)
    styLst.append('*k')
    tBar = t1[-1]+(t2[0]-t1[-1])/2
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = '{} {} {}'.format(siteNo, shortName, code)
        xTS = [x[:, ic] for x in dataLst]
        axplot.plotTS(axP[k], t, xTS, styLst=styLst, tBar=tBar,
                      legLst=['opt1', 'opt2', 'obs'])
        axP[k].set_title(title)

figplot.clickMap(funcMap, funcPoint)
