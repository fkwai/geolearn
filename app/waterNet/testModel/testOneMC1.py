
import copy
import collections
from gc import collect
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

from hydroDL.model import waterNetTestC
import importlib

importlib.reload(waterNetTestC)
importlib.reload(crit)

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff', '00955']
mtdY = ['skip', 'skip']
varXC = gageII.varLstEx
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
siteNo = '09196500'
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
model = waterNetTestC.Wn0110C1(nh, len(varXC), nr)
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
    mDict = copy.deepcopy(dict(model.state_dict()))

    qP, cP = model(xT, xcT)
    lossQ = lossFun(qP, yT[nr-1:, :, 0])
    lossC = lossFun(cP, yT[nr-1:, :, 1])
    loss = lossQ+lossC
    # with torch.autograd.detect_anomaly():
    optim.zero_grad()
    loss.backward()
    optim.step()
    # mDict['fcR.0.weight'].sum()
    # model.state_dict()['fcR.0.weight'].sum()
    print(kk, lossQ.item(), lossC.item())
    b = False
    for name, p in model.named_parameters():
        if p.isnan().any():
            print(kk, name)
            b = True
    if b:
        print(kk, 'break2')
        model.load_state_dict(mDict)
        break

# d1 = collections.OrderedDict(a=[1])
d1 = dict(a=[1])
d2 = copy.deepcopy(d1)
d1['a'].append(2)
d1['a'] = 2
d1
d2

t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()

qOut, cOut = model(xP, xcP)
qP = qOut.detach().cpu().numpy()
cP = cOut.detach().cpu().numpy()
lossQ = lossFun(qOut, yT[nr-1:, :, 0])
lossC = lossFun(cOut, yT[nr-1:, :, 1])
print(lossQ.item(), lossC.item())

k = 0
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(t[nr-1:], qP[:, k], '-r')
axes[0].plot(t, y[:, k, 0], '-k')
axes[1].plot(t[nr-1:], cP[:, k], '-r')
axes[1].plot(t, y[:, k, 1], '*k')
fig.show()


# load LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, indS, :]
yO = y[:, :, 0]
sd = 500
utils.stat.calNash(yL[sd:, :], yO[sd:, :])
utils.stat.calRmse(yL[sd:, :], yO[sd:, :])
utils.stat.calNash(qP[sd:, :], yO[sd+nr-1:, :])
utils.stat.calRmse(qP[sd:, :], yO[sd+nr-1:, :])

utils.stat.calNash(cP[sd:, :], y[sd+nr-1:, :, 1])
utils.stat.calCorr(cP[sd:, :], y[sd+nr-1:, :, 1])
utils.stat.calRmse(cP[sd:, :], y[sd+nr-1:, :, 1])

x = xP
xc = xcP
nt = x.shape[0]
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
nh = model.nh
nr = model.nr
Sf = torch.zeros(ns, nh).cuda()
Ss = torch.zeros(ns, nh).cuda()
Sg = torch.zeros(ns, nh).cuda()
xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
w = model.fcW(xc)
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, model.wLst)
gL = gL*2
kg = kg/10
ga = torch.softmax(model.DP(ga), dim=1)
v = model.fcT(xT)
[vi, ve, vm] = sepPar(v, nh, model.vLst)
vi = F.hardsigmoid(v[:, :, :nh]*2)
ve = ve*2
wR = model.fcR(xc)
vf = torch.arccos((T1+T2)/(T2-T1))/3.1415
vf[T1 >= 0] = 0
vf[T2 <= 0] = 1
Ps = P*vf
Pla = P*(1-vf)
Pl = Pla[:, :, None]*vi
Ev = E[:, :, None]*ve
Q1T = torch.zeros(nt, ns, nh).cuda()
Q2T = torch.zeros(nt, ns, nh).cuda()
Q3T = torch.zeros(nt, ns, nh).cuda()
for k in range(nt):
    qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
    Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
    H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
    qp = torch.relu(kp*(H-gL))
    qs = ks*torch.minimum(H, gL)
    Ss = H-qp-qs
    qso = qs*(1-gp)
    qsg = qs*gp
    qg = kg*(Sg+qsg)+qb
    Sg = Sg-qg
    Q1T[k, :, :] = qp
    Q2T[k, :, :] = qso
    Q3T[k, :, :] = qg
    if (qg < 0).any():
        break
r = torch.relu(wR[:, :nh*nr])
Q1R = convTS(Q1T, r)
Q2R = convTS(Q2T, r)
Q3R = convTS(Q3T, r)
outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)

# concentrations
xTc = torch.cat([T1[:, :, None], T2[:, :, None],
                 torch.tile(xc, [nt, 1, 1])], dim=-1)
wC = model.fcTC(xTc)
[cp, cs, cg] = sepPar(wC, nh, model.cLst)
C1T = Q1T*cp
C2T = Q2T*cs
C3T = Q3T*cg
C1R = convTS(C1T, r)
C2R = convTS(C2T, r)
C3R = convTS(C3T, r)
outC = torch.sum((C1R+C2R+C3R)*ga, dim=2)/outQ

a1 = cp.detach().cpu().numpy()
a2 = cs.detach().cpu().numpy()
a3 = cg.detach().cpu().numpy()
fig, axes = plt.subplots(3, 1)
axes[0].plot(t, a1[:, 0, :])
axes[1].plot(t, a2[:, 0, :])
axes[2].plot(t, a3[:, 0, :])
fig.show()
