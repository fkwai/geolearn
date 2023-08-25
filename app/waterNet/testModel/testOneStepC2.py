
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
codeLst = ['00618', '00915', '00945', '00955']
nc = len(codeLst)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
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
siteNo = '04063700'
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
model = waterNetTestC.Wn0110C2(nh, len(varXC), nr, nc)
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()

# random subset
model.train()
# for kk in range(100):
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


x = xT
xc = xcT
nt = x.shape[0]
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
nh = model.nh
nr = model.nr
Sf = torch.zeros(ns, nh).cuda()
Ss = torch.zeros(ns, nh).cuda()
Sg = torch.zeros(ns, nh).cuda()
w = model.fcW(xc)
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, model.wLst)
gL = gL*2
qb = qb+1e-5
# kg = kg/10
ga = torch.softmax(model.DP(ga), dim=1)
# ga = torch.softmax(ga, dim=1)
xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
v = model.fcT(xT)
[vi, ve, vm] = sepPar(v, nh, model.vLst)
vi = F.hardsigmoid(v[:, :, :nh]*2)
ve = ve*2
wc = model.fcC(xc)
[eqs, eqg] = sepPar(wc, nh*nc, model.cLst)
eqs = eqs.view(ns, nh, nc).permute(-1, 0, 1)*10
eqg = eqg.view(ns, nh, nc).permute(-1, 0, 1)*10
Cs = torch.zeros(nc, ns, nh).cuda()
Cg = eqg*qb/kg
xTC = torch.cat([T1[:, :, None], T2[:, :, None],
                 torch.tile(xc, [nt, 1, 1])], dim=-1)
vc = model.fcCT(xTC)
[rs, rg] = sepPar(vc, nh*nc, model.cLst)
rs = rs.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
rg = rg.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
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
C2T = torch.zeros(nc, nt, ns, nh).cuda()
C3T = torch.zeros(nc, nt, ns, nh).cuda()
for k in range(nt):
    qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
    Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
    H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
    qp = torch.relu(kp*(H-gL))
    qs = ks*torch.minimum(H, gL)
    Ss = H-qp-qs
    Cs = (Cs+rs[:, k, :, :]*eqs*Ss)/(1+ks+rs[:, k, :, :])
    qsg = qs*gp
    qg = kg*(Sg+qsg)+qb
    Sg = (1-kg)*(Sg+qsg)-qb
    Hg = Sg+qb/kg
    Cg = (Cg+rg[:, k, :, :]*eqg*Hg+Cs*ks*gp)/(1+kg+rg[:, k, :, :])
    Q1T[k, :, :] = qp
    Q2T[k, :, :] = qs*(1-gp)
    Q3T[k, :, :] = qg
    C2T[:, k, :, :] = Cs*ks*(1-gp)
    C3T[:, k, :, :] = Cg*kg
r = torch.relu(wR[:, :nh*nr])
Q1R = convTS(Q1T, r)
Q2R = convTS(Q2T, r)
Q3R = convTS(Q3T, r)
outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)
outCLst = list()
for k in range(nc):
    C2R = convTS(C2T[k, ...], r)
    C3R = convTS(C3T[k, ...], r)
    temp = torch.sum((C2R+C3R)*ga, dim=2)/outQ
    outCLst.append(temp)
outC = torch.stack(outCLst, dim=-1)

lossQ = lossFun(outQ, yT[nr-1:, :, 0])
loss = lossQ
lossCLst = list()
for k in range(nc):
    lossC = lossFun(outC[:, :, k], yT[nr-1:, :, k+1])
    lossCLst.append(lossFun(outC[:, :, k], yT[nr-1:, :, k+1]))
    loss = loss+lossC
# with torch.autograd.detect_anomaly():
optim.zero_grad()
loss.backward()
optim.step()

ind = 50
qO = outQ.detach().cpu().numpy()
cO = outC.detach().cpu().numpy()
yO = yT.detach().cpu().numpy()
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(qO[:, ind], '-r')
axes[0].plot(yO[nr-1:, ind, 0], '-k')
axes[1].plot(cO[:, ind], '-r')
axes[1].plot(yO[nr-1:, ind, 1], '*k')
fig.show()

# model.state_dict()
