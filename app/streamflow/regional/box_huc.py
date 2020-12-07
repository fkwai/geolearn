import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn, transform

dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
subsetLst = ['HUC{:02d}'.format(k+1) for k in range(18)]

# global model
nashLst1 = list()
rmseLst1 = list()
corrLst1 = list()
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
for subset in subsetLst:
    indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[subset]]
    nash1 = utils.stat.calNash(yP[indT:, indS, 0], yO[indT:, indS, 0])
    rmse1 = utils.stat.calRmse(yP[indT:, indS, 0], yO[indT:, indS, 0])
    corr1 = utils.stat.calCorr(yP[indT:, indS, 0], yO[indT:, indS, 0])
    nashLst1.append(nash1)
    rmseLst1.append(rmse1)
    corrLst1.append(corr1)

# local model
nashLst2 = list()
rmseLst2 = list()
corrLst2 = list()
for subset in subsetLst:
    testSet = subset
    outName = '{}-{}-B10'.format(dataName, subset)
    yP, ycP = basinFull.testModel(
        outName, DM=dm, batchSize=20, testSet=testSet,reTest=True)
    yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
    nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
    rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
    corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
    # nash2 = utils.stat.calNash(yP[:indT, :, 0], yO[:indT, :, 0])
    # rmse2 = utils.stat.calRmse(yP[:indT, :, 0], yO[:indT, :, 0])
    # corr2 = utils.stat.calCorr(yP[:indT, :, 0], yO[:indT, :, 0])
    nashLst2.append(nash2)
    rmseLst2.append(rmse2)
    corrLst2.append(corr2)

# plot box
matLst = [nashLst1, nashLst2]
label1 = subsetLst
label2 = ['CONUS', 'Local']
dataBox = list()
for k in range(len(subsetLst)):
    temp = list()
    temp.append(matLst[0][k])
    temp.append(matLst[1][k])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()
