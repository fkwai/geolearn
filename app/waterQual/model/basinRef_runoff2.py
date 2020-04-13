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

wqData = waterQuality.DataModelWQ('basinRef')
figFolder = os.path.join(kPath.dirWQ, 'basinRef')

# compare of opt1-4
outLst = ['basinRef-rq-pQ-F50', 'basinRef-q-pQ-F50',
          'basinRef-r-pQ-F50', 'basinRef-opt2']
trainSet = 'pQ-F50'
testSet = 'pQ-L50'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData, ep=400)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData, ep=400)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    pLst1.append(p1)
    pLst2.append(p2)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)

codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    for errMatLst, train in zip([errMatLst1, errMatLst2], ['train', 'test']):
        codeLst = codePdf[codePdf.group == group].index.tolist()
        indLst = [wqData.varC.index(code) for code in codeLst]
        labLst1 = [codePdf.loc[code]['shortName'] +
                   '\n'+code for code in codeLst]
        labLst2 = ['R+Q', 'Q', 'R', 'target Q']
        dataBox = list()
        for ic in indLst:
            temp = list()
            for errMat in errMatLst:
                temp.append(errMat[:, ic, 1])
            dataBox.append(temp)
        title = '{} correlation of {} group'.format(train, group)
        figName = 'box_{}_{}_runoff'.format(train, group)
        fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
        fig.suptitle(title)
        fig.show()
        fig.savefig(os.path.join(figFolder, figName))

# why Q confused some basins

# plot
codeSel = ['00410', '00955']
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf
errMat = errMatLst2[3]-errMatLst2[1]


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = 'D(R) of {} {}'.format(shortName, code)
        axplot.mapPoint(axM[k], lat, lon,
                        errMat[:, ic, 1], s=12, title=title)
    figP, axP = plt.subplots(len(codeSel)+1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    tBar = t1[-1]+(t2[0]-t1[-1])/2
    t = np.concatenate([t1, t2])
    # plot Q
    tq = pd.date_range(t[0], t[-1])
    tempQ = usgs.readStreamflow(siteNo)
    dfQ = pd.DataFrame({'date': tq}).set_index('date').join(tempQ)
    axplot.plotTS(axP[0], dfQ.index.values, [dfQ['00060_00003'].values],
                  tBar=tBar, styLst=['--b'])
    # plot C
    k = 1
    for code in codeSel:
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = '{} {} {}'.format(siteNo, shortName, code)
        xTS = list()
        xTS.append(np.concatenate([pLst1[1][ind1, ic], pLst2[1][ind2, ic]]))
        xTS.append(np.concatenate([pLst1[3][ind1, ic], pLst2[3][ind2, ic]]))
        xTS.append(np.concatenate([o1[ind1, ic], o2[ind2, ic]]))
        axplot.plotTS(axP[k], t, xTS, styLst=['--b', '--r', '*k'],
                      legLst=['w/ Q', 'w/o Q', 'obs'], tBar=tBar)
        axP[k].set_title(title)
        k = k+1


figplot.clickMap(funcMap, funcPoint)

figM.show()
figP.show()


importlib.reload(figplot)
