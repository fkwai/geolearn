
import scipy
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

# wqData = waterQuality.DataModelWQ('basinRef')
# outName = 'basinRef-first50-opt1'
wqData = waterQuality.DataModelWQ('HBN')
outName = 'HBN-first50-opt1'
trainSet = 'first50'
testSet = 'last50'
p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
errMat1 = wqData.errBySite(p1, subset=trainSet)
errMat2 = wqData.errBySite(p2, subset=testSet)

indTrain = wqData.subset[trainSet]
ncAry = np.ndarray([len(wqData.varC)])
for code in wqData.varC:
    ic = wqData.varC.index(code)
    indC = np.where(~np.isnan(wqData.c[indTrain, ic]))[0]
    ncAry[ic] = indC.shape[0]
fig, ax = plt.subplots(1, 1)
ax.plot(ncAry, np.nanmean(errMat1-errMat2, axis=0)[:, 1], '*')
fig.show()


info = wqData.info.loc[wqData.subset[trainSet].tolist()].reset_index()
siteNoLst = wqData.info.siteNo.unique()
ncMat = np.full([len(siteNoLst), len(wqData.varC)], 0)
for i, siteNo in enumerate(siteNoLst):
    indS = info[info['siteNo'] == siteNo].index.values
    for k, code in enumerate(wqData.varC):
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[indS, ic]))[0]
        ncMat[i, k] = indC.shape[0]

fig, ax = plt.subplots(1, 1)
ax.plot(ncMat, errMat1[:, :, 1], '*')
fig.show()

codeLst = ['00955', '00665']
ic = [wqData.varC.index(code) for code in codeLst]
fig, ax = plt.subplots(1, 1)
ax.plot(ncMat[:, ic], errMat2[:, ic, 1], '*')
fig.show()

scipy.stats.spearmanr(ncMat, errMat2[:, :, 1])
