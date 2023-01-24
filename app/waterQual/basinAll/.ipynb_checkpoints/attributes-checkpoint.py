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

wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)
ep = 300
outName = 'basinRef-Yeven-opt2'
trainSet = 'Yodd'
testSet = 'Yeven'
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=ep)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=ep)
errMatC1 = wqData.errBySiteC(
    ycP1, varC=master['varYC'], subset=trainSet,  rmExt=True)
errMatC2 = wqData.errBySiteC(
    ycP2, varC=master['varYC'], subset=testSet, rmExt=True)


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


dropColLst = ['STANAME', 'WR_REPORT_REMARKS',
              'ADR_CITATION', 'SCREENING_COMMENTS']
dfX = gageII.readData(siteNoLst=siteNoLst).drop(columns=dropColLst)
dfX = gageII.updateCode(dfX)
unitConv = 0.3048**3*365*24*60*60/1000**2
groupLst = [['00010', '00095', '00400', '80154', '70303', '00660'],
            ['00665', '00618', '00600', '00605', '71846', '00681'],
            ['00915', '00925', '00935', '00930', '00940', '00945'],
            ['00955', '00410', '00405', '00300', '00950', '00440']]

# area vs error
attr = dfX['DRAIN_SQKM'].values
attr = np.log(attr)
fig, axes = plt.subplots(4, 6)
for j in range(4):
    for i in range(6):
        code = groupLst[j][i]
        ic = wqData.varC.index(code)
        ind = np.where((countMat[:, ic, 0] > 20) &
                       (countMat[:, ic, 1] > 20))[0]
        err = errMatC2[ind, ic, 1]
        axes[j, i].plot(attr[ind], err, '*')
        axes[j, i].set_title(code)
# plt.tight_layout()
fig.show()

