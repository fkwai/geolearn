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
dataName = 'chloride'
wqData = waterQuality.DataModelWQ(dataName)
outLst = ['chloride-Yodd-ntn', 'chloride-Yodd']
# outLst = ['sulfateNE-Yodd-ntn-silica', 'sulfateNE-Yodd-silica']
trainSet = 'Yodd'
testSet = 'Yeven'
# outLst = ['sulfateNE-Yeven-ntn', 'sulfateNE-Yeven']
# trainSet = 'Yeven'
# testSet = 'Yodd'

errMatLst1, errMatLst2, ypLst1, ypLst2 = [list() for x in range(4)]
for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(
        outName, trainSet, wqData=wqData, ep=100, reTest=True)
    yP2, ycP2 = basins.testModel(
        outName, testSet, wqData=wqData, ep=100, reTest=True)
    ypLst1.append(ycP1)
    ypLst2.append(ycP2)

ypLst1[1]=ypLst1[1][~np.isnan(ypLst1[0])]
ypLst2[1]=ypLst2[1][~np.isnan(ypLst2[0])]
for k in range(2):
    errMatC1 = wqData.errBySiteC(ypLst1[k], subset=trainSet, varC=master['varYC'])
    errMatC2 = wqData.errBySiteC(ypLst2[k], subset=testSet, varC=master['varYC'])
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)



fig, axes = plt.subplots(1, 2)
indC = wqData.varC.index('00940')
ind = wqData.subset[testSet].tolist()
# ind=wqData.subset[trainSet].tolist()
axes[0].plot(wqData.c[ind, indC], ypLst2[1], '*')
axes[1].plot(wqData.c[ind, indC], ypLst2[0], '*')
fig.show()


# time series
siteNoLst = wqData.siteNoLst
fig, axes = plt.subplots(3, 1)
for k in range(3):
    siteNo = siteNoLst[k]
    info1 = wqData.info.iloc[wqData.subset[trainSet]].reset_index()
    info2 = wqData.info.iloc[wqData.subset[testSet]].reset_index()
    t1 = info1['date'].values
    t2 = info2['date'].values
    indS1 = info1[info1['siteNo'] == siteNo].index.values
    indS2 = info2[info2['siteNo'] == siteNo].index.values
    t = wqData.info[wqData.info['siteNo'] == siteNo]['date'].values
    v = wqData.c[wqData.info['siteNo'] == siteNo]
    axplot.plotTS(axes[k], t1[indS1], [yp[indS1]for yp in ypLst1],
                  cLst='rm', styLst='**', legLst=['train w/o ntn', 'train w/ ntn'])
    axplot.plotTS(axes[k], t2[indS2], [yp[indS2]for yp in ypLst2],
                  cLst='bc', styLst='**', legLst=['test w/o ntn', 'test w/ ntn'])
    axplot.plotTS(axes[k], t, v, cLst='k', styLst='*', legLst=['obs'])
fig.show()
