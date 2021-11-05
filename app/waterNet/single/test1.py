
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from hydroDL.model import waterNetTest
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

siteNoLst = ['03144000']
dataName = 'temp'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, sdStr='1982-01-01', edStr='2018-12-31')
trainSet = 'WYB10'
testSet = 'WYA10'
DF.saveSubset('WYB10', ed='2009-10-01')
DF.saveSubset('WYA10', sd='2009-10-01')

DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx']
mtdX = ['skip' for k in range(4)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup2 = DM2.getData()


# model
nh = 16
model = waterNetTest.WaterNetSingle(nh)
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
# random subset
model.train()
for kk in range(100):
    batchSize = [365, 100]
    [rho, nbatch] = batchSize
    sizeLst = trainBasin.getSize(dataTup)
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
    iS = np.random.randint(0, ns, [nbatch])
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch, nx], np.nan)
    xcTemp = np.full([nbatch, nxc], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    ycTemp = np.full([nbatch, nyc], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if xc is not None:
        xcTemp = xc[iS, :]
    if yc is not None:
        ycTemp = yc[iS, :]
    xT = torch.from_numpy(xTemp).float().cuda()
    xcT = torch.from_numpy(xcTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    ycT = torch.from_numpy(ycTemp).float().cuda()
    model.zero_grad()
    yP = model(xT)
    loss = lossFun(yP[:, :, None], yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())

model.eval()

t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
yOut = model(xP)
yP = yOut.detach().cpu().numpy()
lossFun(yOut[:, :, None], yT)
model.zero_grad()

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(t, yP[:, k], '-r')
ax.plot(t, y[:, k], '-k')
fig.show()

utils.stat.calNash(yP, y[:, :, 0])
utils.stat.calCorr(yP, y[:, :, 0])
