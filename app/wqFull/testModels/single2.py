import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
from hydroDL.model import rnn, crit, trainBasin, test

# look at those sites : '07144100'
# 00915 - 07144100  09352900  01435000
siteNo = '402114105350101'
codeLst = ['00915', '00600']

# plot data
varLst = ['00915', 'runoff']
df = dbBasin.readSiteTS(siteNo, varLst=varLst, freq='D')
fig, axes = plt.subplots(len(varLst), 1, figsize=(12, 3))
for k, code in enumerate(varLst):
    axplot.plotTS(axes[k], df.index, df[code].values, cLst='k')
fig.show()

# load data
sd = '1982-01-01'
ed = '2018-12-31'
siteNoLst = [siteNo]
DM = dbBasin.DataModelFull.new('test', siteNoLst, sdStr=sd, edStr=ed)

# define inputs
varX = gridMET.varLst+GLASS.varLst+ntn.varLst + \
    ['datenum', 'sinT', 'cosT'] + ['runoff']
# varX = ['datenum', 'sinT', 'cosT'] + ['runoff']
varXC = None
varY = codeLst
# varY = ['runoff']
varYC = None
varTup = [varX, varXC, varY, varYC]
dataTupRaw = DM.extractData(varTup, 'all', '1982-01-01', '2018-12-31')
dataTup, statTup = DM.transIn(dataTupRaw, varTup)
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
sizeLst = trainBasin.getSize(dataTup)
[nx, nxc, ny, nyc, nt, ns] = sizeLst

# train
importlib.reload(test)
model = test.LSTM(nx+nxc, ny+nyc, 256).cuda()
model
lossFun = crit.RmseLoss().cuda()
optim = torch.optim.Adadelta(model.parameters())
model, optim, lossEp = trainBasin.trainModel(
    dataTup, model, lossFun, optim, batchSize=[365, 100],
    nEp=100, logFile='logFile')

# test
dataTupRaw = DM.extractData(varTup, 'all', '1982-01-01', '2018-12-31')
dataTup = DM.transIn(dataTupRaw, varTup, statTup=statTup)
sizeLst = trainBasin.getSize(dataTup)
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
x = dataTup[0]
xc = dataTup[1]
ny = sizeLst[2]
yOut, ycOut = trainBasin.testModel(model, x, xc, ny, batchSize=20)
yP = DM.transOut(yOut, statTup[2], varY)

# plot
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10')
saveFile = os.path.join(dirWRTDS, siteNo)
df = pd.read_csv(saveFile, index_col=None).set_index('date')
# output
yT = dataTupRaw[2]
t = DM.t
nc = len(varY)
fig, axes = plt.subplots(nc, 1, figsize=(12, 3))
for k in range(nc):
    axplot.plotTS(axes[k], df.index.values.astype('datetime64[D]'),
                  df[codeLst[k]].values, styLst='-', cLst='b')
    axplot.plotTS(axes[k], t, [yP[:, 0, k], yT[:, 0, k]],
                  styLst='-*', cLst='rk')
fig.show()


# # transformed
# fig, axes = plt.subplots(nc, 1, figsize=(12, 3))
# yIn = dataTup[2]
# for k in range(nc):
#     axplot.plotTS(axes[k], t, [yOut[:, 0, k], yIn[:, 0, k]],
#                   styLst='-*', cLst='rk')

# fig.show()

# fig, ax = plt.subplots(1, 1, figsize=(12, 3))
# ax.plot(dataTupRaw[2][:, :, 0],'*')
# fig.show()
