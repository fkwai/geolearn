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

outName = '{}-B10'.format(dataName)
nashLst1 = list()
nashLst2 = list()
rmseLst1 = list()
rmseLst2 = list()
corrLst1 = list()
corrLst2 = list()
testSet = 'all'
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

# plot box
label1 = ['nash', 'rmse', 'corr']
label2 = ['train', 'test']
dataBox = [[nash1, nash2], [rmse1, rmse2], [corr1, corr2]]
# dataBox = [[nash1, rmse1, corr1], [nash2, rmse2, corr2]]
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()
