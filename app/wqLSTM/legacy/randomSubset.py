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

siteNoLst = ['07144100', '09352900', '01435000']
codeLst = ['00915', '00600']

# load data
sd = '1982-01-01'
ed = '2018-12-31'
DM = dbBasin.DataModelFull.new('test', siteNoLst, sdStr=sd, edStr=ed)

# define inputs
# varX = gridMET.varLst+['datenum', 'sinT', 'cosT']
varX = ['datenum', 'sinT', 'cosT', 'runoff']
varXC = None
varY = codeLst
varYC = None
varTup = [varX, varXC, varY, varYC]
dataTupRaw = DM.extractData(varTup, 'all', sd, ed)
dataTup, statTup = DM.transIn(dataTupRaw, varTup)
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
sizeLst = trainBasin.getSize(dataTup)
[nx, nxc, ny, nyc, nt, ns] = sizeLst
xTensor, yTensor = trainBasin.subsetRandom(dataTup, [365, 100], sizeLst)

dataLst = dataTup
batchSize = [365, 100]
# rewrite subset
[x, xc, y, yc] = dataLst
[rho, nbatch] = batchSize
[nx, nxc, ny, nyc, nt, ns] = sizeLst
iS = np.random.randint(0, ns, [nbatch])
matB = ~np.isnan(y[rho:, :, :])

s1 = np.sum(matB, axis=2)
s2 = np.sum(matB, axis=(0, 2))
wT = s1/s2
wS = s2 / np.sum(s2)
iS = np.random.choice(ns, nbatch, p=wS)
iT = np.zeros(nbatch)
for k in range(nbatch):
    iT[k] = np.random.choice(nt-rho, p=wT[:, iS[k]])+rho

# plot weights
fig, axes = plt.subplots(ns, 1, figsize=(12, 3))
for k in range(ns):
    axes[k].plot(np.arange(nt), wT[:, k])
    ax2 = axes[k].twinx()
    ax2.plot(np.arange(nt), y[:, k, 1], 'r*')
fig.show()

df = dbBasin.readSiteTS(siteNoLst[0], varLst=varY+varX, freq='D',
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
