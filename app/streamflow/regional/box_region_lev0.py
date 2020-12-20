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


caseLst = ['08', '09']


dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
nash0 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
rmse0 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
corr0 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
bias0 = utils.stat.calBias(yP[indT:, :, 0], yO[indT:, :, 0])

nashLst = list()
rmseLst = list()
corrLst = list()
biasLst = list()


for case in caseLst:
    testSet = 'Eco'+case
    nashLstTemp = list()
    rmseLstTemp = list()
    corrLstTemp = list()
    biasLstTemp = list()

    # global model
    indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[testSet]]
    nashLstTemp.append(nash0[indS])
    rmseLstTemp.append(rmse0[indS])
    corrLstTemp.append(corr0[indS])
    biasLstTemp.append(bias0[indS])

    trainLst = [case[:2]]
    outLst = ['{}-Eco{}-B10-gs'.format(dataName, x)
              for x in trainLst]
    for outName in outLst:
        yP, ycP = basinFull.testModel(
            outName, DM=dm, batchSize=20, testSet=testSet)
        yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
        nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
        rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
        corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
        bias2 = utils.stat.calBias(yP[indT:, :, 0], yO[indT:, :, 0])
        nashLstTemp.append(nash2)
        rmseLstTemp.append(rmse2)
        corrLstTemp.append(corr2)
        biasLstTemp.append(bias2)
    nashLst.append(nashLstTemp)
    rmseLst.append(rmseLstTemp)
    corrLst.append(corrLstTemp)
    biasLst.append(biasLstTemp)


# plot box
label1 = caseLst
label2 = ['CONUS', 'lev0']
dataBox = rmseLst
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4))
fig.show()
