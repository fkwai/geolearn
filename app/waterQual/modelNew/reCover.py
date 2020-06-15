import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear, wqRela
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from hydroDL.model import rnn, crit
import os


siteNo = '401733105392404'
# siteNo = '01364959'
codeLst = ['00915', '00940', '00955']
saveDir = r'C:\Users\geofk\work\waterQuality\modelNew2'
sFile = os.path.join(saveDir, '{}_S.csv'.format(siteNo))
bFile = os.path.join(saveDir, '{}_B.csv'.format(siteNo))
zFile = os.path.join(saveDir, '{}_Z.csv'.format(siteNo))
lFile = os.path.join(saveDir, '{}_L.csv'.format(siteNo))

matS = pd.read_csv(sFile, header=None).values
matB = pd.read_csv(bFile, header=None).values
matZ = pd.read_csv(zFile, header=None).values
matL = pd.read_csv(lFile, header=None).values

varX = gridMET.varLst
varY = ['00060']
dfX = waterQuality.readSiteX(siteNo, varX)
dfY = waterQuality.readSiteY(siteNo, varY)
mtdX = waterQuality.extractVarMtd(varX)
mtdY = waterQuality.extractVarMtd(varY)
# mtdY[0] = 'norm'
# mtdX[0] = 'norm'
normX, statX = transform.transInAll(dfX.values, mtdX)
dfXN = pd.DataFrame(data=normX, index=dfX.index, columns=dfX.columns)
normY, statY = transform.transInAll(dfY.values, mtdY)
dfYN = pd.DataFrame(data=normY, index=dfY.index, columns=dfY.columns)


p = dfXN.values[:, 0]
rho = 365
nt = len(p)
matQA = np.ndarray([nt-rho, rho+1])
for k in range(rho, nt):
    i = k-rho
    q = p[i:k]*matS[:, i]
    b = matB[:, i]
    qA = np.add.accumulate(q)*b[1]+b[0]
    # qA = np.add.accumulate(q)+b[0]
    matQA[i, 1:] = qA
    matQA[i, 0] = b[0]

outQA = transform.transOut(matQA, mtdY[0], statY[0])
outL = transform.transOut(matL, mtdY[0], statY[0])

t = dfY.index.values[rho:]
tBar = np.datetime64('2000-01-01')
fig, ax = plt.subplots(1, 1)

data = [dfY.values[rho:], outQA[:, 365], outQA[:, 180], outQA[:, 0]]
axplot.plotTS(ax, t, data, tBar=tBar,
              legLst=['obs', '> 0 day', '> 180 day', 'base flow'], styLst='----', cLst='kbcg')

fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(dfY.values[rho:], '-k')
ax.plot(outL[0, rho:].flatten(), '-r')
fig.show()


tLst = ['2000-01-01', '2000-01-02', '2000-03-01', '2000-06-01', '2000-09-01']
cLst = 'rgbmc'
i1 = np.where(t == np.datetime64('2000-01-01'))[0][0]
fig, ax = plt.subplots(1, 1)
for k, ts in enumerate(tLst):
    i = np.where(t == np.datetime64(ts))[0][0]
    ax.plot(matS[:, i].T, '-'+cLst[k], label='gate {}'.format(ts))
ax.legend()
fig.show()

dLst = [0, 179, 364]
dataLst = [matS[i, :] for i in dLst]
legLst = ['gate {} day'.format(i+1) for i in dLst]
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, t, dataLst, styLst='---', cLst='rgb', legLst=legLst)
fig.show()


y = dfY['00060'].values
x = dfX['pr'].values
nt = len(y)
mapMat = np.ndarray(365)
for i in range(1, 366):
    [a, b] = utils.rmNan([x[:-i], y[i:]])
    mapMat[i-1] = np.corrcoef(a, b)[0, 1]
fig, ax = plt.subplots(1, 1)
ax.plot(mapMat)
ax.legend()
fig.show()
