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

yPLst = list()
yOLst = list()

yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]

# for case in caseLst:
case = caseLst[0]
testSet = 'Eco'+case
indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[testSet]]
yPLst.append(yP[:, indS, 0])
yOLst.append(yO[:, indS, 0])

trainLst = [case[:2]]
outLst = ['{}-Eco{}-B10-gs'.format(dataName, x)
          for x in trainLst]
for outName in outLst:
    yP, ycP = basinFull.testModel(
        outName, DM=dm, batchSize=20, testSet=testSet)
    yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
    yPLst.append(yP[:, :, 0])
    yOLst.append(yO[:, :, 0])

ns = yP.shape[1]
k = np.random.randint(ns)
fig, ax = plt.subplots(figsize=(12, 4))
dataLst = [yP[:, k] for yP in yPLst]+[yOLst[0][:, k]]
labLst = ['CONUS', 'lev0', 'lev1', 'lev2', 'obs']
cLst = 'rmgbk'

axplot.plotTS(ax, dm.t, dataLst, styLst='-----', cLst=cLst, legLst=labLst)
fig.show()

rmse0 = utils.stat.calRmse(yPLst[0][indT:, :], yPLst[3][indT:, :])
rmse1 = utils.stat.calRmse(yPLst[1][indT:, :], yPLst[3][indT:, :])
rmse2 = utils.stat.calRmse(yPLst[2][indT:, :], yPLst[3][indT:, :])


# plot box
label1 = caseLst
label2 = ['CONUS', 'lev0', 'lev1', ]
dataBox = [[rmse0, rmse1, rmse2]]
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgk', label1=label1,
                      label2=label2, figsize=(6, 4))
fig.show()
