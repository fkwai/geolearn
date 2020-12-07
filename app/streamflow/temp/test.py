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


# importlib.reload(dbBasin.io)
importlib.reload(dbBasin.dataModel)
importlib.reload(dbBasin)
importlib.reload(basinFull)


dataName = 'Q90ref'
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
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='br', label1=label1,
                      label2=label2, figsize=(6, 4), yRange=[0, 1])
fig.show()

# plot map
siteNoLst = dm.siteNoLst
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLst)
dfCrd = gageII.updateCode(dfCrd)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
figM, axM = plt.subplots(3, 1, figsize=(6, 8))
axplot.mapPoint(axM[0], lat, lon, nash2, s=16)
axM[0].set_title('test nash')
axplot.mapPoint(axM[1], lat, lon, rmse2, s=16)
axM[1].set_title('test rmse')
axplot.mapPoint(axM[2], lat, lon, corr2, s=16)
axM[2].set_title('test corr')
figM.show()
