
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

from hydroDL.model import waterNetGlobal
import importlib

importlib.reload(waterNetGlobal)
importlib.reload(crit)

siteNoLst = ['02212600']
dataName = 'temp'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, sdStr='1982-01-01', edStr='2018-12-31')
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst)
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')

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

trainSet = 'B10'
testSet = 'A10'
DM = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup = DM.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup2 = DM2.getData()


# model
nh = 16
model = waterNetGlobal.WaterNet2(nh, len(varXC))
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters(), lr=0.1)
# optim = torch.optim.Rprop(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

# random subset
model.train()
for kk in range(50):
    batchSize = [1000, 100]
    sizeLst = trainBasin.getSize(dataTup)
    [x, xc, y, yc] = dataTup
    [rho, nbatch] = batchSize
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
    yP = model(xT, xcT)
    loss = lossFun(yP[:, :, None], yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.item())

model.eval()

t = DF.getT(trainSet)
[x, xc, y, yc] = dataTup

t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
yOut = model(xP, xcP)
yP = yOut.detach().cpu().numpy()
lossFun(yOut[:, :, None], yT)
model.zero_grad()

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(t, yP[:, k], '-r')
ax.plot(t, y[:, k], '-k')
ax.twinx().plot(DF.t, DF.f[:, 0, DF.varF.index('LAI')], '-b')

# ax.plot(t, x[:, k,0])
fig.show()

nash = utils.stat.calNash(yP, y[:, :, 0])
corr = utils.stat.calCorr(yP, y[:, :, 0])

w = model.fc(xcP)
gm = torch.exp(w[:, :nh])+1
ge = torch.sigmoid(w[:, nh:nh*2])
go = torch.sigmoid(w[:, nh*2:nh*3])
gl = torch.exp(w[:, nh*3:nh*4])
gb = torch.sigmoid(w[:, nh*5:nh*6])
qb = w[:, -1]
kb = torch.sigmoid(w[:, nh*6:nh*7])


# all years
fig, ax = plt.subplots(1, 1)
# ax.plot(DF.t, DF.f[:, 0, DF.varF.index('pr')], '-r')
ax.plot(DF.t, DF.f[:, 0, DF.varF.index('LAI')], '-r')
ax.twinx().plot(DF.t, DF.q[:, 0, 1])
fig.show()

# test years
t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
fig, axes = plt.subplots(2, 1)
ax = axes[0]
ax.plot(t, x[:, 0, 0], '-b')
ax.twinx().plot(t, y[:, 0, 0], '-r')
ax = axes[1]
ax.plot(t, x[:, 0, 1], '-b')
ax.twinx().plot(t, x[:, 0, 2], '-r')
fig.show()
