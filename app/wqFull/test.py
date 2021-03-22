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
from hydroDL.app import waterQuality as wq


# importlib.reload(dbBasin.io)
importlib.reload(dbBasin.dataModel)
importlib.reload(dbBasin)
importlib.reload(basinFull)


dataName = 'sbY30N5'
dm = dbBasin.DataModelFull(dataName)
outName = '{}-B10'.format(dataName)
master = basinFull.loadMaster(outName)
varY = master['varY']
testSet = 'all'
sd = '1979-01-01'
ed = '2020-01-01'

yP, ycP = basinFull.testModel(outName, DM=dm, batchSize=20, testSet=testSet)
yO, ycO = basinFull.getObs(outName, testSet, DM=dm)

indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
importlib.reload(utils.stat)
siteNoLst = dm.siteNoLst
corrMat = np.ndarray([len(siteNoLst), len(varY), 2])
for k in range(len(varY)):
    corr1 = utils.stat.calCorr(yP[:indT, :, k], yO[:indT, :, k])
    corr2 = utils.stat.calCorr(yP[indT:, :, k], yO[indT:, :, k])
    corrMat[:, k, 0] = corr1
    corrMat[:, k, 1] = corr2

# load previous result
if True:
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, varY[1:])
    corrMatO, rmseMatO = wq.dictErr(dictLSTM, dictWRTDS, dictObs, varY[1:])

# plot box
label1 = varY[1:]
label2 = ['train', 'test']
dataBox = list()
for k in range(1, len(varY)):
    # dataBox.append([corrMat[:, k, 1], corrMatO[:, k, 1]])
    dataBox.append([corrMat[:, k, 1], corrMatO[:, k-1, 1]])

fig = figplot.boxPlot(dataBox, widths=0.5, cLst='br', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()
