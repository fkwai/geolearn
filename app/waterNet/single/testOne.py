
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
siteNo = '03187500'
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
# model = waterNetTest.WaterNet1115(nh, len(varXC))
model = waterNetTest.WaterNet1115(nh, 1, len(varXC))
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
    loss = lossFun(yP[:, :, None], yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss.item())
    w = model.fc(xcT)
    # print(w[0, :])


model.eval()

t = DF.getT(trainSet)
[x, xc, y, yc] = dataTup

t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
yOut, (q1Out, q2Out, q3Out) = model(xP, xcP, outQ=True)
yP = yOut.detach().cpu().numpy()
q1P = q1Out.detach().cpu().numpy()
q2P = q2Out.detach().cpu().numpy()
q3P = q3Out.detach().cpu().numpy()

lossFun(yOut[:, :, None], yT)
model.zero_grad()

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(t, yP[:, k], '-r')
ax.plot(t, y[:, k], '-k')
fig.show()


# k = 0
# fig, ax = plt.subplots(1, 1)
# ax.plot(t, yP[:, k], '-r')
# ax.plot(t, y[:, k], '-k')
# ax.twinx().plot(t, x[:,0,:])
# fig.show()

x = xP
xc = xcP
P, E, T1, T2, LAI = [x[:, :, k] for k in range(5)]
nt, ns = P.shape
Ta = (T1+T2)/2
rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
rP[T1 >= 0] = 1
rP[T2 <= 0] = 0
Ps = P*(1-rP)
Pl = P*rP
S0 = torch.zeros(ns, nh).cuda()
S1 = torch.zeros(ns, nh).cuda()
Sv = torch.zeros(ns, nh).cuda()
S2 = torch.zeros(ns, nh).cuda()
S3 = torch.zeros(ns, nh).cuda()
Yout = torch.zeros(nt, ns).cuda()
w = model.fc(xc)
xcT = torch.cat([LAI[:, :, None], T1[:, :, None],
                 T2[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
v = model.fcT(xcT)
gm = torch.exp(w[:, :nh])+1
k1 = torch.sigmoid(w[:, nh:nh*2])
k2 = torch.sigmoid(w[:, nh*2:nh*3])
k23 = torch.sigmoid(w[:, nh*3:nh*4])
k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
gl = torch.exp(w[:, nh*5:nh*6])*2
ga = torch.softmax(w[:, nh*6:nh*7], dim=1)
qb = torch.relu(w[:, nh*7:nh*8])
vi = F.hardsigmoid(v[:, :, :nh])
vk = F.hardsigmoid(v[:, :, nh:nh*2])
ve1 = torch.relu(v[:, :, nh*2:nh*3])
ve2 = torch.relu(v[:, :, nh*3:nh*4])
vm = torch.exp(v[:, :, nh*4:nh*5])

Pl1 = Pl[:, :, None]*(1-vi)
Pl2 = Pl[:, :, None]*vi
Ev1 = E[:, :, None]*ve1
Ev2 = E[:, :, None]*ve2


# load LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, indS, :]
yO = y[:, :, 0]
sd = 0
utils.stat.calNash(yL[sd:, :], yO[sd:, :])
utils.stat.calRmse(yL[sd:, :], yO[sd:, :])
utils.stat.calNash(yP[sd:, :], yO[sd:, :])
utils.stat.calRmse(yP[sd:, :], yO[sd:, :])

temp = vi.detach().cpu().numpy()[:, 0, :]
a = ga.detach().cpu().numpy()[0, :]
x = xP.detach().cpu().numpy()[:, 0, :]
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(t, temp*a)
# axes[0].plot(t, temp)
axes[1].plot(t, yP, '-r')
axes[1].plot(t, yL, '-b')
# ax = axes[1].twinx()
# ax.plot(t, np.abs(yP-yO), '-r')
# ax.plot(t, np.abs(yL-yO), '-b')
axes[1].plot(t, y[:, k], '-k')
axes[2].plot(t, q1P[:, 0, :]*a)
fig.show()

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(t, x[:,  [0, 2, 3, 4]])
axes[1].plot(t, yP, '-r')
axes[1].plot(t, yL, '-b')
axes[1].plot(t, y[:, k], '-k')
ax = axes[1].twinx()
ax.plot(t, np.abs(yP-yO)-np.abs(yL-yO), '--k')

axes[2].plot(t, np.abs(yP-yO), '-r')
axes[2].plot(t, np.abs(yL-yO), '-b')
fig.show()
