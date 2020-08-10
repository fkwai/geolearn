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
dataName = 'sulfateNE'
wqData = waterQuality.DataModelWQ(dataName)
outLst = ['sulfateNE-Yodd-ntn', 'sulfateNE-Yodd']
# outLst = ['sulfateNE-Yodd-ntn-silica', 'sulfateNE-Yodd-silica']
trainSet = 'Yodd'
testSet = 'Yeven'
# outLst = ['sulfateNE-Yeven-ntn', 'sulfateNE-Yeven']
# trainSet = 'Yeven'
# testSet = 'Yodd'

errMatLst1, errMatLst2, ypLst1, ypLst2 = [list() for x in range(4)]
for outName in outLst:
    master = basins.loadMaster(outName)
    yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=200)
    yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=200)
    errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
    errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
    errMatLst1.append(errMatC1)
    errMatLst2.append(errMatC2)
    ypLst1.append(ycP1)
    ypLst2.append(ycP2)

varC = master['varYC']
fig, axes = plt.subplots(1, 2)
indC1 = wqData.varC.index('00935')
indC2 = master['varYC'].index('00935')
ind = wqData.subset[testSet].tolist()
# ind=wqData.subset[trainSet].tolist()
axes[0].plot(wqData.c[ind, indC1], ypLst2[1][:, indC2], '*')
axes[1].plot(wqData.c[ind, indC1], ypLst2[0][:, indC2], '*')
fig.show()


fig, axes = plt.subplots(1, 2)
indF = wqData.varF.index('ph')
indC = wqData.varC.index('00945')
axes[0].plot(wqData.c[:, indC], wqData.f[-1, :, indF], '*')
indF = wqData.varF.index('SO4')
axes[1].plot(wqData.c[:, indC], wqData.f[-1, :, indF], '*')
fig.show()

indF = wqData.varF.index('ph')
_, ind = utils.rmNan([wqData.c[:, indC], wqData.f[-1, :, indF]])
np.corrcoef(wqData.c[ind, indC], wqData.f[-1, ind, indF])

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
    v = wqData.c[wqData.info['siteNo'] == siteNo][:, 0]
    axplot.plotTS(axes[k], t1[indS1], [yp[indS1, 0] for yp in ypLst1],
                  cLst='rm', styLst='**', legLst=['train w/o ntn', 'train w/ ntn'])
    axplot.plotTS(axes[k], t2[indS2], [yp[indS2, 0]for yp in ypLst2],
                  cLst='bc', styLst='**', legLst=['test w/o ntn', 'test w/ ntn'])
    axplot.plotTS(axes[k], t, v, cLst='k', styLst='*', legLst=['obs'])
fig.show()
