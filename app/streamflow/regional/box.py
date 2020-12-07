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


# subsetLst = [
#     '09', '0903', '090303',
#     '09','0904', '090402',
#     '08', '0803', '080305',
# ]
case = '090402'
dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)

trainLst = [case[:6], case[:4], case[:2]]
outLst = ['{}-{}-B10'.format(dataName, x)
          for x in trainLst]+['{}-B10'.format(dataName)]
nashLst1 = list()
nashLst2 = list()
rmseLst1 = list()
rmseLst2 = list()
corrLst1 = list()
corrLst2 = list()
testSet = case
for outName in outLst:
    yP, ycP = basinFull.testModel(
        outName, DM=dm, batchSize=20, testSet=testSet)
    yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
    indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
    nash1 = utils.stat.calNash(yP[:indT, :, 0], yO[:indT, :, 0])
    nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
    rmse1 = utils.stat.calRmse(yP[:indT, :, 0], yO[:indT, :, 0])
    rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
    corr1 = utils.stat.calCorr(yP[:indT, :, 0], yO[:indT, :, 0])
    corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
    nashLst1.append(nash1)
    nashLst2.append(nash2)
    rmseLst1.append(rmse1)
    rmseLst2.append(rmse2)
    corrLst1.append(corr1)
    corrLst2.append(corr2)

# plot box
label1 = ['nash', 'rmse', 'corr']
label2 = ['lev2', 'lev1', 'lev0', 'CONUS']
dataBox = [nashLst2, rmseLst2, corrLst2]
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()
