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
from hydroDL.model import rnn, crit, trainBasin

siteNo = '06800500'
codeLst = ['00915', '00925']

# plot data
df = dbBasin.readSiteTS(siteNo, varLst=codeLst, freq='D')
nc = len(codeLst)
fig, axes = plt.subplots(nc, 1, figsize=(12, 3))
for k, code in enumerate(codeLst):
    axplot.plotTS(axes[k], df.index, df[code].values, cLst='k')
fig.show()

# load data
sd = '1982-01-01'
ed = '2018-12-31'
siteNoLst = [siteNo]
DM = dbBasin.DataModelFull.new('test', siteNoLst, sdStr=sd, edStr=ed)

# define inputs
# varX = gridMET.varLst+['datenum', 'sinT', 'cosT']
varX = ['datenum', 'sinT', 'cosT']
varXC = None
varY = ['runoff']+codeLst
varYC = None
varTup = [varX, varXC, varY, varYC]
dataTupRaw = DM.extractData(varTup, 'all', sd, ed)
dataTup, statTup = DM.transIn(dataTupRaw, varTup)
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
sizeLst = trainBasin.getSize(dataTup)
[nx, nxc, ny, nyc, nt, ns] = sizeLst
xTensor, yTensor = trainBasin.subsetRandom(dataTup, [365, 100], sizeLst)


df = dbBasin.readSiteTS(siteNo, varLst=varY+varX, freq='D',
                        sd=np.datetime64(sd),
                        ed=np.datetime64(ed))
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
k = 1
j = 15
yTemp = yTensor.detach().cpu().numpy()
xTemp = xTensor.detach().cpu().numpy()
datenum = transform.transOutAll(
    xTemp[:, :, -3:-2], ['norm'],  [statTup[0][-3]])
ax.plot(dataTupRaw[0][:, 0, -3], dataTup[2][:, 0, k], 'k*')
# ax.plot(df['datenum'], np.log(df[varTup[2][k]]+1), 'k*')
ax.plot(datenum[:, j, 0], yTemp[:, j, k], 'r.')
fig.show()
