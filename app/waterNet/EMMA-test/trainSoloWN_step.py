
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS

# extract data
codeLst = ['00600', '00605', '00618', '71846']
siteNoLst = ['04193500']
dataName = 'temp'
# DF = dbBasin.DataFrameBasin.new(
#     dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
# DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
# DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(dataName)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
mtdX = ['skip' for k in range(4)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['stan' for code in codeLst]
varXC = gageII.varLstEx
mtdXC = ['skip' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()


# check data plot
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(DM1.t, DM1.y[:, 0, :], labelLst=labelLst)
ax = axes[0].twinx()
ax.plot(DM1.t, DM1.x[:, 0, 0], 'b')
ax.invert_yaxis()
fig.show()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nc = len(codeLst)

# wrap up data
[rho, nbatch] = batchSize
iS = np.random.randint(0, ns, [nbatch])
iT = np.random.randint(0, nt-rho, [nbatch])
xTemp = np.full([rho, nbatch, nx], np.nan)
yTemp = np.full([rho, nbatch, ny], np.nan)
if x is not None:
    for k in range(nbatch):
        xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
if y is not None:
    for k in range(nbatch):
        yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
xT = torch.from_numpy(xTemp).float().cuda()
yT = torch.from_numpy(yTemp).float().cuda()

# parameters for a single model
DP = nn.Dropout()
wR = Parameter(torch.randn(nh*nr).cuda())
wLst = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
        'exp', 'relu', 'skip']
w = Parameter(torch.randn(nh*len(wLst)).cuda())
vLst = ['skip', 'relu', 'exp']
fcT = nn.Sequential(
    nn.Linear(6, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nh*len(vLst))).cuda()
v = fcT(xT)
cp = Parameter(torch.randn(nh, nc).cuda())
cs = Parameter(torch.randn(nh, nc).cuda())
cg = Parameter(torch.randn(nh, nc).cuda())
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

# forward
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, wLst)
gL = gL**2
kg = kg/10
ga = torch.softmax(DP(ga), dim=-1)
[vi, ve, vm] = sepPar(v, nh, vLst)
vi = F.hardsigmoid(vi*2)
ve = ve*2
rf = torch.relu(wR)

# forward
x = xT
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
Sf = torch.zeros(ns, nh).cuda()
Ss = torch.zeros(ns, nh).cuda()
Sg = torch.zeros(ns, nh).cuda()
Ps, Pl, Ev = WaterNet0119.forwardPreQ(P, E, T1, T2, vi, ve)
QpT = torch.zeros(nt, ns, nh).cuda()
QsT = torch.zeros(nt, ns, nh).cuda()
QgT = torch.zeros(nt, ns, nh).cuda()
for k in range(nt):
    qp, qs, qg, Sf, Ss, Sg = WaterNet0119.forwardStepQ(
        Sf, Ss, Sg, Ps[k, :, None], Pl[k, :, :],
        Ev[k, :, :], vm[k, :, :], kp, ks, kg, gL, gp, qb)
    QpT[k, :, :] = qp
    QsT[k, :, :] = qs
    QgT[k, :, :] = qg
QpR = convTS(QpT, rf)
QsR = convTS(QsT, rf)
QgR = convTS(QgT, rf)
Qout = torch.sum((QpR+QsR+QgR)*ga, dim=-1)


CpR = torch.matmul(QpR*ga, cp)
CsR = torch.matmul(QsR*ga, cs)
CgR = torch.matmul(QgR*ga, cg)
Cout = CpR+CsR+CgR
yP = torch.cat([Qout[..., None], Cout], dim=-1)
loss = lossFun(yP[:, :, None], yT[nr-1:, :, :])