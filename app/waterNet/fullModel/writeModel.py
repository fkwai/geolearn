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

# load data
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

# find out rho
t = DF.getT(trainSet)
tW = np.datetime64('1980-10-01')
rhoW = np.where(t == tW)[0][0]
rho = (5, 365, rhoW)

# step by step model
import argparse

self = argparse.Namespace()
nf = x.shape[-1]
nh = 4
ng = xc.shape[-1]
nr = 5
hs = 256
dr = 0.5
import hydroDL
from hydroDL.model.waterNet.modelFull import defineFCNN
import importlib

importlib.reload(hydroDL.model.waterNet.modelFull)

self.nf = nf
self.nh = nh
self.ng = ng
self.nr = nr
self.hs = hs
self.dr = dr
self.rho_short, self.rho_long, self.rho_warmup = rho
self.initParam(hs=hs, dr=dr)


def initParam(self, hs=256, dr=0.5):
    # gates [kp, ks, kg, gp, gl, qb, ga]
    self.gDict = OrderedDict(
        kp=lambda x: torch.sigmoid(x),  # ponding
        ks=lambda x: torch.sigmoid(x),  # shallow
        kd=lambda x: torch.sigmoid(x),  # deep
        gd=lambda x: torch.sigmoid(x),  # partition of shallow to deep
        gl=lambda x: torch.pow(torch.exp(x), 2),  # effective depth
        qb=lambda x: torch.relu(x),  # baseflow
        ga=lambda x: torch.softmax(x, -1),  # area
        gi=lambda x: F.hardsigmoid(x),  # interception
        ge=lambda x: torch.relu(x),  # evaporation
    )
    self.kDict = dict(
        km=lambda x: torch.exp(x),  # snow melt
    )
    self.FC = nn.Linear(self.ng, hs)
    self.FC_r = nn.Linear(hs, self.nh * self.nr)
    self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
    self.FC_kin = nn.Linear(4, hs)
    self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))


def getParam(self, x, xc):
    f = x[:, :, 2:]
    nt = x.shape[0]
    state = self.FC(xc)
    mask_k = createMask(state, self.dr)
    mask_g = createMask(state, self.dr)
    mask_r = createMask(state, self.dr)
    pK = self.FC_kout(
        DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k)
    )  # check in debug
    pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g))
    pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r))
    paramK = sepParam(pK, self.nh, self.kDict)
    paramG = sepParam(pG, self.nh, self.gDict)
    paramR = pR
    return paramK, paramG, paramR


def reset_parameters(self):
    for layer in self.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def initState(self, ns):
    Sf = torch.zeros(ns, self.nh)
    Ss = torch.zeros(ns, self.nh)
    Sg = torch.zeros(ns, self.nh)
    if torch.cuda.is_available():
        Sf = Sf.cuda()
        Ss = Ss.cuda()
        Sg = Sg.cuda()
    return Sf, Ss, Sg


# fake functions
self.initParam = initParam
self.getParam = getParam
self.initState = initState

importlib.reload(bucket)
# forward
self.initParam(self, self.hs, self.dr)
nt = x.shape[0]
ns = x.shape[1]
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Sf, Ss, Sd = self.initState(self, ns)
paramK, paramG, paramR = self.getParam(self, x, xc)

# general forcings
Ps, Pl = func.divideP(Prcp, T1, T2)
Ps = Ps.unsqueeze(-1)
Pl = Pl.unsqueeze(-1)
Evp = Evp.unsqueeze(-1)

k = 0
Sf, qf = bucket.snow(Sf, Ps[k, ...], paramK['km'][k, ...])
Is = qf + Pl[k, ...] * paramG['gi'] - Evp[k, ...] * paramG['ge']
Ss, qp, qsA = bucket.shallow(Ss, Is, L=paramG['gl'], k1=paramG['kp'], k2=paramG['ks'])
qs = qsA * (1 - paramG['gd'])
Id = qsA * paramG['gd']
Sd, qd = bucket.deep(Sd, Id, k=paramG['kd'], baseflow=paramG['qb'])


nt = x.shape[0]
ns = x.shape[1]
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Sf, Ss, Sd = self.initState(self,ns)
paramK, paramG, paramR = self.getParam(self,x,xc)
Ps, Pl = func.divideP(Prcp, T1, T2)
global qf

def forwardLong(k):
    Sf, qf = bucket.snow(Sf, Ps[k, ...], paramK['km'][k, ...])

def forwardShort(k):
    Is = qf + Pl[k, ...] * paramG['gi'] - Evp[k, ...] * paramG['ge']
    Ss, qp, qsA = bucket.shallow(
        Ss, Is, L=paramG['gl'], k1=paramG['kp'], k2=paramG['ks']
    )
    qs = qsA * (1 - paramG['gd'])
    Id = qsA * paramG['gd']
    Sd, qd = bucket.deep(Sd, Id, k=paramG['kd'], baseflow=paramG['qb'])

forwardLong(10)