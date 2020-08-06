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
dataName = 'refWeek'
wqData = waterQuality.DataModelWQ(dataName)
outLst = ['refWeek-Yodd', 'refWeek-Yodd-ntn']
trainSet = 'Yodd'
testSet = 'Yeven'
# outLst = ['sulfateNE-Yeven-ntn', 'sulfateNE-Yeven']
# trainSet = 'Yeven'
# testSet = 'Yodd'

errMatLst1, errMatLst2, ypLst1, ypLst2 = [list() for x in range(4)]
for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=500)
    yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=500)
    errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
    errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)
    ypLst1.append(ycP1)
    ypLst2.append(ycP2)


# figure out number of sample
siteNoLst = wqData.info['siteNo'].unique().tolist()
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

# plot box
codePdf = usgs.codePdf
groupLst = [['00010', '00095', '00400', '80154', '70303', '00660',
             '00665', '00618', '00600', '00605', '71846', '00681'],
            ['00915', '00925', '00935', '00930', '00940', '00945',
             '00955', '00410', '00405', '00300', '00950', '00440']]
strLst = ['physical and nutrient variables', 'inorganics variables']
for k in range(2):
    codeLst = groupLst[k]
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['train', 'train-ntn', 'test', 'test-ntn']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst1+errMatLst2:
            ind = np.where((countMat[:, ic, 0] > 10) &
                           (countMat[:, ic, 1] > 10))[0]
            temp.append(errMat[ind, ic, 1])
        dataBox.append(temp)
    fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                          label2=labLst2, figsize=(12, 4), yRange=[0, 1])
    title = 'correlation of {}'.format(strLst[k])
    fig.suptitle(title)
    fig.show()

# plot map
codeSel = ['00945', '00935']
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf
figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
for k, code in enumerate(codeSel):
    ic = wqData.varC.index(code)
    shortName = codePdf.loc[code]['shortName']
    title = 'correlation difference of {} {}'.format(shortName, code)
    ind = np.where((countMat[:, ic, 0] > 20) &
                   (countMat[:, ic, 1] > 20))[0]
    data = errMatLst2[1][ind, ic, 1]-errMatLst2[0][ind, ic, 1]
    axplot.mapPoint(axM[k], lat[ind], lon[ind], data, s=12, vRange=[-1, 1])
    axM[k].set_title(title)
figM.show()


# ['0143400680', '01434021', '01434025']
siteNo = '01434025'
indS = siteNoLst.index(siteNo)
indC = wqData.varC.index('00945')
errMatLst2[0][indS, indC]
errMatLst2[1][indS, indC]

errMatLst2[0][50, indC]
errMatLst2[1][50, indC]

fig, axes = plt.subplots(1, 2)
ind = wqData.subset[testSet].tolist()
# ind=wqData.subset[trainSet].tolist()
indC = wqData.varC.index('00945')
axes[0].plot(wqData.c[ind, indC], ypLst2[0][:, indC], '*')
axes[1].plot(wqData.c[ind, indC], ypLst2[1][:, indC], '*')
fig.show()

ind = wqData.subset[trainSet].tolist()

(a, b), ind2 = utils.rmNan([wqData.c[ind, indC], ypLst1[0][:, indC]])
np.corrcoef(a, b)
