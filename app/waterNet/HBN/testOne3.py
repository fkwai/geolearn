
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

dataName = 'QN90'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm']
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

# extract subset
siteNo = '08057410'
siteNoLst = DF.getSite(trainSet)
indS = siteNoLst.index(siteNo)
dataLst1 = list()
dataLst2 = list()
for dataLst, dataTup in zip([dataLst1, dataLst2], [dataTup1, dataTup2]):
    for data in dataTup:
        if data is not None:
            if data.ndim == 3:
                data = data[:, indS:indS+1, :]
            else:
                data = data[indS:indS+1, :]
        dataLst.append(data)
dataTup1 = tuple(dataLst1)
dataTup2 = tuple(dataLst2)

# model
nh = 16
model = waterNetGlobal.WaterNet3(nh, 1, len(varXC))
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
# optim = torch.optim.Rprop(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()
w = model.fc(xcP)
print(w[0, :])


# random subset
model.train()
for kk in range(100):
    batchSize = [1000, 100]
    sizeLst = trainBasin.getSize(dataTup1)
    [x, xc, y, yc] = dataTup1
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
    w = model.fc(xcT)
    print(w[0, :])


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
# ax.twinx().plot(DF.t, DF.f[:, 0, DF.varF.index('LAI')], '-b')

# ax.plot(t, x[:, k,0])
fig.show()

nash = utils.stat.calNash(yP, y[:, :, 0])
corr = utils.stat.calCorr(yP, y[:, :, 0])

w = model.fc(xcP)
P, E, T1, T2, LAI = [xP[:, :, k] for k in range(5)]
nt, ns = P.shape
xcT = torch.cat([LAI[:, :, None], torch.tile(xcP, [nt, 1, 1])], dim=-1)
v = model.fcT(xcT)
gm = torch.exp(w[:, :nh])+1
ge = torch.sigmoid(w[:, nh:nh*2])*2
go = torch.sigmoid(w[:, nh*2:nh*3])
gl = torch.exp(w[:, nh*3:nh*4]*2)
ga = torch.softmax(w[:, nh*4:nh*5], dim=1)
gb = torch.sigmoid(w[:, nh*5:nh*6])
qb = torch.relu(w[:, -1])/nh
kb = torch.sigmoid(w[:, nh*6:nh*7])/10
vi = torch.relu(v[:, :, :nh])

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
