import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, func
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
from hydroDL.model.waterNet.func import convTS, sepParam
from hydroDL.model.dropout import createMask, DropMask
from collections import OrderedDict
from hydroDL.model.waterNet.modelFull import WaterNet0313
from hydroDL.model import crit
import matplotlib.pyplot as plt
import time
import os

# load data
saveDir = r'/oak/stanford/schools/ees/kmaher/Kuai/waterQuality/waterNet/'

code = '00955'
dataName = '{}-{}'.format(code, 'B200')
DF = dbBasin.DataFrameBasin(dataName)

label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'sph']
mtdX = (
    ['skip' for k in range(2)]
    + ['scale' for k in range(2)]
    + ['norm' for k in range(2)]
    + ['skip' for k in range(2)]
)
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'B15'
testSet = 'A15'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC
)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC
)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

(xP, xcP, yP, ycP) = dataTup1
x = torch.from_numpy(xP).float()
xc = torch.from_numpy(xcP).float()
y = torch.from_numpy(yP).float()

# find out rho
t = DF.getT(trainSet)
tW = np.datetime64('1980-10-01')
rhoW = np.where(t == tW)[0][0]
rho = (5, 365, rhoW)
nf = x.shape[-1]
nh = 16
ng = xc.shape[-1]
nr = 5
hs = 256
dr = 0.5
nd = 366

# fake self class
self = torch.nn.Module()
self.__init__()

# def __init__(self, nf, ng, nh, nr, rho=(5, 365, 0), hs=256, dr=0.5):
self.nf = nf
self.nh = nh
self.ng = ng
self.nr = nr
self.hs = hs
self.dr = dr
self.nd = nd
self.rho_short, self.rho_long, self.rho_warmup = rho

# def initParam(self, hs=256, dr=0.5):
self.gDict = OrderedDict(
    gk=lambda x: torch.exp(x) / 100,  # curve parameter
    gl=lambda x: torch.exp(x) * 100,  # effective depth
    qb=lambda x: torch.relu(x) / 10,  # baseflow
    ga=lambda x: torch.softmax(x, -1),  # area
    gi=lambda x: F.hardsigmoid(x) / 2,  # interception
    ge=lambda x: torch.relu(x),  # evaporation
    gc=lambda x: torch.exp(x) * 10,  # lim concentraion
    gr=lambda x: torch.sigmoid(x),  # age-reaction rate
)
self.kDict = dict(
    km=lambda x: torch.exp(x),  # snow melt
    ki=lambda x: torch.relu(x),  # interception
    ke=lambda x: torch.exp(x),  # evaporation
)
self.FC = nn.Linear(self.ng, hs)
self.FC_r = nn.Linear(hs, self.nh * (self.nr + 1))
self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
self.FC_kin = nn.Linear(4, hs)
self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))

# getParam
f = x[:, :, 2:]  # T1, T2, Rad and Hum
nt = x.shape[0]
state = self.FC(xc)
mask_k = createMask(state, self.dr)
mask_g = createMask(state, self.dr)
mask_r = createMask(state, self.dr)
pK = self.FC_kout(
    DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k, self.training)
)
pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g, self.training))
pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r, self.training))
paramK = sepParam(pK, self.nh, self.kDict)
paramG = sepParam(pG, self.nh, self.gDict)
paramR = func.onePeakWeight(pR, self.nh, self.nr)


# start
nt = x.shape[0]
ns = x.shape[1]
# initState
Sf = torch.zeros(ns, self.nh)
D = torch.zeros(ns, self.nh, self.nd)
if torch.cuda.is_available():
    Sf = Sf.cuda()
    D = D.cuda()
storage = (Sf, D)

Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Ps, Pl = func.divideP(Prcp, T1, T2)
Ps = Ps.unsqueeze(-1)
Pl = Pl.unsqueeze(-1)
Evp = Evp.unsqueeze(-1)
input = [Ps, Pl, Evp]
param = [paramK, paramG, paramR]

storage = (Sf, D)
# forward all time steps
SfLst, DLst = [], []
with torch.no_grad():
    for iT in range(nt):
        storage, flux = bucket.stepSAS(iT, storage, input, param)
        Sf, Ss = storage
        SfLst.append(Sf)
        DLst.append(D)

# training
nbatch, rho = 100, 365
iS = torch.randint(0, ns, [nbatch])
iT = torch.randint(self.rho_warmup, nt - rho, [nbatch])
Sf_sub = torch.FloatTensor(nbatch, self.nh)
D_sub = torch.FloatTensor(nbatch, self.nh, self.nd)
Ps_sub = torch.FloatTensor(rho, nbatch, 1)
Pl_sub = torch.FloatTensor(rho, nbatch, 1)
Evp_sub = torch.FloatTensor(rho, nbatch, 1)
q_obs = torch.FloatTensor(rho, nbatch)
if torch.cuda.is_available():
    Sf_sub = Sf_sub.cuda()
    D_sub = D_sub.cuda()
    Ps_sub = Ps_sub.cuda()
    Pl_sub = Pl_sub.cuda()
    Evp_sub = Evp_sub.cuda()
    q_obs = q_obs.cuda()
# minibatch of input and state
for k in range(nbatch):
    Sf_sub[k, :] = SfLst[iT[k]][iS[k], :]
    D_sub[k, :, :] = DLst[iT[k]][iS[k], :, :]
    Ps_sub[:, k, :] = Ps[iT[k] : iT[k] + rho, iS[k], :]
    Pl_sub[:, k, :] = Pl[iT[k] : iT[k] + rho, iS[k], :]
    Evp_sub[:, k, :] = Evp[iT[k] : iT[k] + rho, iS[k], :]
    q_obs[:, k, :] = y[iT[k] : iT[k] + rho, iS[k]]

# minibatch of param
paramK_sub = dict()
for key in self.kDict:
    paramK_sub[key] = torch.FloatTensor(rho, nbatch, self.nh)
    if torch.cuda.is_available():
        paramK_sub[key] = paramK_sub[key].cuda()
    for k in range(nbatch):
        paramK_sub[key][:, k, :] = paramK[key][iT[k] : iT[k] + rho, iS[k], :]
paramG_sub = paramG.copy()
for key in paramG_sub.keys():
    paramG_sub[key] = paramG_sub[key][iS, :]
paramR_sub = paramR[:, iS, :]

input = [Ps_sub, Pl_sub, Evp_sub]
storage = (Sf_sub, D_sub)
param = (paramK_sub, paramG_sub, paramR_sub)
QLst = list()
DLst = list()
for k in range(rho):
    storage, flux = bucket.stepSAS(k, storage, input, param)
    QLst.append(flux[1])
    DLst.append(storage[1])

# convert to Q and C
Q = torch.stack(QLst, dim=0)
D = torch.stack(DLst, dim=0)
q_bucket = torch.sum(Q, dim=-1)
q_pred = torch.sum(q_bucket * paramG_sub['ga'], dim=-1)
lossFun = crit.LogLoss2D()
loss = lossFun(q_pred, q_obs)
loss.backward()
optim.step()

max(Q[Q < 0])
