
from hydroDL.app.waterQuality import WRTDS
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import os
import json
import numpy as np
import pandas as pd
import time
from hydroDL import kPath, utils
from hydroDL.data import usgs, transform, dbBasin
import statsmodels.api as sm
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dataName = 'weathering'
DF = dbBasin.DataFrameBasin(dataName)
t0 = np.datetime64('1982-01-01', 'D')
t1 = np.datetime64('2010-01-01', 'D')
t2 = np.datetime64('2018-12-31', 'D')
codeLst = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
siteNoLst = DF.siteNoLst
trainSet = 'rmRT20'
testSet = 'pkRT20'

# load WRTDS
yO = DF.extractT(codeLst)
yW = np.ndarray(yO.shape)
for k, siteNo in enumerate(siteNoLst):
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-D', 'weathering-B10')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    df.index = df.index.values.astype('datetime64[D]')
    ind = (df.index >= t0) & (df.index <= t2)
    yW[:, k, :] = df.loc[ind][codeLst].values


yW2 = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)

siteNo = '01184000'
code = '00915'
indS = siteNoLst.index(siteNo)
indC = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
tsData = [yW2[:, indS, indC], yO[:, indS, indC]]
axplot.plotTS(ax, DF.t, tsData)
fig.show()

# # new version
# trainSet = 'B10'
# testSet = 'A10'
# sn = 1e-5
# q = DF.q[:, :, 0]
# logQ = np.log(q+sn)
# tt = pd.to_datetime(DF.t)
# yr = tt.year.values
# t = yr+tt.dayofyear.values/365
# sinT = np.sin(2*np.pi*t)
# cosT = np.cos(2*np.pi*t)
# obs = DF.extractT(codeLst)
# Y1 = DF.extractSubset(obs, trainSet)
# Y2 = DF.extractSubset(obs, testSet)

# fitAll = False
# indS = 0
# indC = 0

# ind1 = np.where(mask1[:, indS])[0]
# ind2 = np.where(mask2[:, indS])[0]
# X = np.stack([logQ[:, indS], yr, sinT, cosT]).T
# Y = obs[:, indS, indC]


# aa = d1.X[:, 0, 1]
# fig, axes = plt.subplots(2, 1)
# axes[0].plot(d1.X[:, 0, 1])
# axes[1].plot(d1.X[:, 0, 2])
# fig.show()
