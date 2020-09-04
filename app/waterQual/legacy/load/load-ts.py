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

wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)
wqData.c = wqData.c * wqData.q[-1, :, 0:1]

outName = 'basinRef-Yodd-opt1'
trainSet = 'Yodd'
testSet = 'Yeven'

master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
ycP1 = ycP1*yP1[-1, :, :]
ycP2 = ycP2*yP2[-1, :, :]
errMatC1 = wqData.errBySiteC(
    ycP1, varC=master['varYC'], subset=trainSet,  rmExt=True)
errMatC2 = wqData.errBySiteC(
    ycP2, varC=master['varYC'], subset=testSet, rmExt=True)

# figure out number of sample
info1 = wqData.subsetInfo(trainSet)
info2 = wqData.subsetInfo(testSet)
dataTrain = wqData.extractSubset(trainSet)
dataTest = wqData.extractSubset(testSet)
ycT1 = dataTrain[3]
ycT2 = dataTest[3]
nc = ycT1.shape[1]
siteNoLst = wqData.info['siteNo'].unique().tolist()
countMat = np.full([len(siteNoLst), nc, 2], 0)
for i, siteNo in enumerate(siteNoLst):
    indS1 = info1[info1['siteNo'] == siteNo].index.values
    indS2 = info2[info2['siteNo'] == siteNo].index.values
    for iC in range(nc):
        countMat[i, iC, 0] = np.count_nonzero(~np.isnan(ycT1[indS1, iC]))
        countMat[i, iC, 1] = np.count_nonzero(~np.isnan(ycT2[indS2, iC]))


# plot
codeSel = ['00955', '00935', '00915']
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
    ind = np.where((countMat[:, ic, 0] > 20) &
                   (countMat[:, ic, 1] > 20))[0]
    indLst.append(ind)
indAll = np.unique(np.concatenate(indLst))
siteNoLstP = [siteNoLst[i] for i in indAll]
outLst = ['basinRef-Yodd-opt1', 'basinRef-Yeven-opt1']


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
    siteNo = siteNoLstP[iP]
    dfPred1, _ = basins.loadSeq(outLst[0], siteNo)
    dfPred2, _ = basins.loadSeq(outLst[1], siteNo)
    sd = np.datetime64('1980-01-01')
    dfQ = waterQuality.readSiteY(siteNo, ['00060'], sd=sd)
    dfC = waterQuality.readSiteY(
        siteNo, codeSel+[code+'_cd' for code in codeSel], sd=sd)
    dfPred1 = dfPred1[dfPred1.index >= sd]
    dfPred2 = dfPred2[dfPred2.index >= sd]
    dfPred1 = dfPred1.multiply(dfPred1['00060'], axis='index')
    dfPred2 = dfPred2.multiply(dfPred2['00060'], axis='index')
    dfC[codeSel] = dfC[codeSel].multiply(dfQ['00060'], axis='index')

    t = dfPred1.index.values.astype(np.datetime64)
    # axplot.plotTS(axP[0], t, [dfPred1['00060'], dfQ['00060']], tBar=tBar,
    #               legLst=['pred-opt1', 'obs'], styLst='--', cLst='br')
    # axP[0].set_title('{} streamflow'.format(siteNo))
    for k, var in enumerate(codeSel):
        shortName = codePdf.loc[var]['shortName']
        title = '{} {} {}'.format(siteNo, shortName, var)
        styLst = ['-', '-', '*', '*', '*', '*']
        legLst = ['model odd', 'model even', 'obs odd',
                  'obs even', 'flag even', 'flag odd']
        yr = dfC.index.year
        c1 = dfC[var].values.copy()
        c2 = dfC[var].values.copy()
        f1 = dfC[var].values.copy()
        f2 = dfC[var].values.copy()
        vf = dfC[var+'_cd'].values
        c1[(vf != 'x') & (vf != 'X')] = np.nan
        c1[(yr % 2 == 0)] = np.nan
        c2[(vf != 'x') & (vf != 'X')] = np.nan
        c2[(yr % 2 == 1)] = np.nan
        f1[(vf == 'x') | (vf == 'X') | (yr % 2 == 0)] = np.nan
        f2[(vf == 'x') | (vf == 'X') | (yr % 2 == 1)] = np.nan
        data = [dfPred1[var].values, dfPred2[var].values, c1, c2, f1, f2]
        axplot.plotTS(axP[k], t, data, styLst=styLst, cLst='bgrmkk',
                      legLst=legLst)
        axP[k].set_title(title)


plt.tight_layout
figplot.clickMap(funcMap, funcPoint)
