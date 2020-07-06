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


outName = 'basinRef-Y8090-opt1'
trainSet = 'Y8090'
testSet = 'Y0010'
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
q1, c1 = basins.getObs(outName, trainSet, wqData=wqData)
q2, c2 = basins.getObs(outName, testSet, wqData=wqData)

# seq test
outLst = ['basinRef-Y8090-opt1', 'basinRef-Y8090-rmF-opt1']
siteNoLst = wqData.info['siteNo'].unique().tolist()
for outName in outLst:
    basins.testModelSeq(outName, siteNoLst, wqData=wqData)

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
codeSel = ['00665', '00660']
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
    figP, axP = plt.subplots(len(codeSel)+1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon[indAll], lat[indAll]


def funcPoint(iP, axP):
    siteNo = siteNoLstP[iP]
    tBar = np.datetime64('2000-01-01')
    dfPred1, _ = basins.loadSeq(outLst[0], siteNo)
    dfPred2, _ = basins.loadSeq(outLst[1], siteNo)
    sd = np.datetime64('1980-01-01')
    dfQ = waterQuality.readSiteY(siteNo, ['00060'], sd=sd)
    dfC = waterQuality.readSiteY(
        siteNo, codeSel+[code+'_cd' for code in codeSel], sd=sd)
    dfPred1 = dfPred1[dfPred1.index >= sd]
    dfPred2 = dfPred2[dfPred2.index >= sd]
    t = dfPred1.index.values.astype(np.datetime64)
    axplot.plotTS(axP[0], t, [dfPred1['00060'], dfQ['00060']], tBar=tBar,
                  legLst=['pred-opt1', 'obs'], styLst='--', cLst='br')
    axP[0].set_title('{} streamflow'.format(siteNo))
    for k, var in enumerate(codeSel):
        shortName = codePdf.loc[var]['shortName']
        title = ' {} {}'.format(shortName, var)
        styLst = ['-', '-', '*', '*']
        vc = dfC[var].values.copy()
        vf = dfC[var+'_cd'].values
        vcf = dfC[var].values.copy()
        vcf[(vf == 'x') | (vf == 'X')] = np.nan
        data = [dfPred1[var].values, dfPred2[var].values, vc, vcf]
        axplot.plotTS(axP[k+1], t, data, tBar=tBar,
                      legLst=['pred', 'pred-rmFlag', 'obs', 'obs-flag'], styLst=styLst, cLst='bgrk')
        axP[k+1].set_title(title)


plt.tight_layout
figplot.clickMap(funcMap, funcPoint)
