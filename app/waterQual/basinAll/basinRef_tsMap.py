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


outName = 'basinRef-first50-opt1'
trainSet = 'first50'
testSet = 'last50'
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
q1, c1 = basins.getObs(outName, trainSet, wqData=wqData)
q2, c2 = basins.getObs(outName, testSet, wqData=wqData)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)


# figure out number of sample
info = wqData.info
siteNoLst = info['siteNo'].unique().tolist()
ycT = wqData.c
nc = ycT.shape[1]
countMat = np.full([len(siteNoLst), nc], 0)
for i, siteNo in enumerate(siteNoLst):
    indS = info[info['siteNo'] == siteNo].index.values
    for iC in range(nc):
        countMat[i, iC] = np.count_nonzero(~np.isnan(ycT[indS, iC]))

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
    ind = np.where(countMat[:, ic] > 100)[0]
    indLst.append(ind)
ind0 = indLst[0]
for ind in indLst:
    ind0 = list(set(ind0).intersection(set(ind)))
siteNoLstP = [siteNoLst[i] for i in ind0]

def funcMap():
    figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = 'correlation of {} {}'.format(shortName, code)
        ind = indLst[k]
        axplot.mapPoint(axM[k], lat[ind], lon[ind], errMatC2[ind, ic, 1], s=12)
        axM[k].set_title(title)
    figP, axP = plt.subplots(len(codeSel)+1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon[ind0], lat[ind0]


def funcPoint(iP, axP):
    siteNo = siteNoLstP[iP]
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    tBar = t1[-1]+(t2[0]-t1[-1])/2
    t = np.concatenate([t1, t2])
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    dfPred = dfPred[dfPred.index >= np.datetime64('1980-01-01')]
    dfObs = dfObs[dfObs.index >= np.datetime64('1980-01-01')]
    t = dfPred.index.values.astype(np.datetime64)
    axplot.plotTS(axP[0], t, [dfPred['00060'], dfObs['00060']], tBar=tBar,
                  legLst=['pred', 'obs'], styLst='--', cLst='br')
    axP[0].set_title('streamflow')
    for k, var in enumerate(codeSel):
        styLst = ['--*', '--*']
        shortName = codePdf.loc[var]['shortName']
        title = ' {} {}'.format(shortName, var)
        [x1, x2], iT = utils.rmNan([dfPred[var].values, dfObs[var].values])
        axplot.plotTS(axP[k+1], t[iT], [x1, x2], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k+1].set_title(title)
    print(countMat[iP, [9, 10]])


plt.tight_layout

figplot.clickMap(funcMap, funcPoint)
