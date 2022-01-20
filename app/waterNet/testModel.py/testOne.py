
from hydroDL.model.waterNet import convTS, sepPar
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

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
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
# siteNo = '09196500'
siteNo = '04063700'
# siteNo = '07148400'
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
nr = 5
model = waterNetTest.WaterNet0119(nh, len(varXC), nr)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.Rprop(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()

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
    loss = lossFun(yP[:, :, None], yT[nr-1:, :, :])
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss.item())


model.eval()


t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
yOut = model(xP, xcP)
yP = yOut.detach().cpu().numpy()

loss = lossFun(yOut[:, :, None], yT[nr-1:, :, :])
model.zero_grad()

# load LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, indS, :]
yO = y[:, :, 0]
sd = 500
utils.stat.calNash(yL[sd:, :], yO[sd:, :])
utils.stat.calRmse(yL[sd:, :], yO[sd:, :])
utils.stat.calNash(yP[sd:, :], yO[sd+nr-1:, :])
utils.stat.calRmse(yP[sd:, :], yO[sd+nr-1:, :])

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(t[nr-1:], yP[:, k], '-r')
ax.plot(t, yL, '-b')
ax.plot(t, y[:, k], '-k')
fig.show()

# check parameters
x = xP
xc = xcP
xcT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
xcT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
v = model.fcT(xcT)
[vi, ve, vm] = sepPar(v, nh, model.vLst)

p = x.detach().cpu().numpy()
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t[nr-1:], yP[:, k], '-r')
axes[0].plot(t, y[:, k], '-k')
axes[1].plot(t, p[:, 0, [2,3]])
fig.show()
