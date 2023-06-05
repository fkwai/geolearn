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

# initState
Sf = torch.zeros(ns, self.nh)
Ss = torch.zeros(ns, self.nh, self.nd)
if torch.cuda.is_available():
    Sf = Sf.cuda()
    Ss = Ss.cuda()

# start
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Ps, Pl = func.divideP(Prcp, T1, T2)
Ps = Ps.unsqueeze(-1)
Pl = Pl.unsqueeze(-1)
Evp = Evp.unsqueeze(-1)
input = [Ps, Pl, Evp]
param = [paramK, paramG, paramR]

iT = 0
storage = (Sf, Ss, Sg)
storage, flux = bucket.step(iT, storage, input, param)

(Sf_new, Ss_new, Sd_new) = storage
(qf, qp, qs, qd) = flux

# storage curve

Sf, Ss, Sd = storage
Pl, Ps, Evp = input

gl = paramG['gl']
gk = paramG['gk']
Sf_new, qf = bucket.snow(Sf, Ps[iT, ...], paramK['km'][iT, ...])
if 'gi' in paramG:
    P = Pl[iT, ...] * paramG['gi']
elif 'ki' in paramK:
    P = Pl[iT, ...] * paramG['ki'][iT, ...]
if 'ge' in paramG:
    E = Evp[iT, ...] * paramG['ge']
elif 'ke' in paramK:
    E = Evp[iT, ...] * paramG['ke'][iT, ...]
Is = qf + P - E


nd = 366

S = torch.zeros(ns, self.nh, nd)
S = torch.ones(ns, self.nh, nd)
D = torch.cumsum(S, -1) / 100

gl = paramG['gl'].unsqueeze(-1)
gk = paramG['gk'].unsqueeze(-1)

D_new = D.clone()
D_new[:, :, :-1] = D[:, :, 1:]
D_total = torch.maximum(D[:,:,-1] + Is,torch.zeros(D[:,:,-1].shape))
D_total_mat=D_total.unsqueeze(-1).repeat(1, 1, nd)
D_new[:, :, -1] = D_total
D_update = torch.minimum(D_new, D_total_mat)

Is[-1,-2]
D_update[-1,-2,:]

k1 = torch.exp(gk * (D_update -gl)) / gk
k1[D_update > gl] = D_update[D_update > gl]
k2 = torch.cat([torch.exp(gk * -gl) / gk, k1[:, :, :-1]], dim=-1)
Q = k1 - k2
Q_cum = torch.cumsum(Q, -1)
D_update = D_update - Q_cum


Q[D > gl] = S_new[D > gl]
S_out = S_new - Q

a = D > gl
b = D > gl.repeat(1, 1, nd)
torch.equal(a, b)


k1[0, 0, :10]
k2[0, 0, :10]
S_new[0, 0, :10]
Q2 = S_new * k1

Q[0, 0, :10]
Q2[0, 0, :10]

torch.exp(-gk * gl).shape
