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

l3Lst = ['080304', '080305', '080401',
         '080503', '090402', '090303']


dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
nash0 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
rmse0 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
corr0 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])

nashLst = list()
rmseLst = list()
corrLst = list()

for case in l3Lst:
    testSet = 'Eco'+case
    nashLstTemp = list()
    rmseLstTemp = list()
    corrLstTemp = list()

    # global model
    indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[testSet]]
    nashLstTemp.append(nash0[indS])
    rmseLstTemp.append(rmse0[indS])
    corrLstTemp.append(corr0[indS])

    trainLst = [case[:2], case[:4], case[:6]]
    outLst = ['{}-Eco{}-B10-gs'.format(dataName, x)
            for x in trainLst]
    for outName in outLst:
        yP, ycP = basinFull.testModel(
            outName, DM=dm, batchSize=20, testSet=testSet)
        yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
        nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
        rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
        corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
        nashLstTemp.append(nash2)
        rmseLstTemp.append(rmse2)
        corrLstTemp.append(corr2)
    nashLst.append(nashLstTemp)
    rmseLst.append(rmseLstTemp)
    corrLst.append(corrLstTemp)


# plot box
label1 = l3Lst
label2 = ['CONUS', 'lev0', 'lev1', 'lev2']
dataBox = nashLst
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()
