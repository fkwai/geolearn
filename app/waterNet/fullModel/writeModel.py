import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model.waterNet import bucket, trans
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np

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

import argparse

self = argparse.Namespace()
self.rho_short, self.rho_long, self.rho_warmup = rho
self.nh = 4
self.ng = 55
self.nr = 3
nh, ng, nr = self.nh, self.ng, self.nr
dr = 0.5


def forwardLong(S, I, paramK):
    Sf, qf = bucket.snow(S, I, paramK)
    return Sf, qf


def forwardShort(S, I, paramK, paramG):
    Sf0, Ss0, Sd0 = S
    qf, Pl, Ev = I
    fm, fi = paramK
    [kp, ks, kg, gr, gL, qb] = paramG
    Ss, qp, qs = bucket.shallow(Ss0, Pl * fi + qf - Ev, (gL, kp, ks))
    Sd, qd = bucket.deep(Sd0, qs * gr, (kg, qb))
    return (Ss, Sd), (qp, qs * (1 - gr), qd)


def initState(ns):
    nh = self.nh
    Sf = torch.zeros(ns, nh)
    Ss = torch.zeros(ns, nh)
    Sg = torch.zeros(ns, nh)
    if torch.cuda.is_available():
        Sf = Sf.cuda()
        Ss = Ss.cuda()
        Sg = Sg.cuda()
    return Sf, Ss, Sg


self.fcR = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Dropout(p=dr),
    nn.Linear(256, nh * nr),
)
# [kp, ks, kg, gp, gl, qb, ga]
self.wLst = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'exp', 'relu', 'skip']
self.fcW = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Dropout(p=dr),
    nn.Linear(256, nh * len(self.wLst)),
)
# [vi,ve,vm]
self.vLst = ['skip', 'relu', 'exp']
self.fcT = nn.Sequential(
    nn.Linear(6 + ng, 256),
    nn.Tanh(),
    nn.Dropout(p=dr),
    nn.Linear(256, nh * len(self.vLst)),
)
self.DP = nn.Dropout(p=dr)

# fake class input
self.initState = initState
self.forwardLong = forwardLong
self.forwardShort = forwardShort

[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, self.wLst)


nt = x.shape[0]
ns = x.shape[1]
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Sf, Ss, Sd = self.initState(ns)
Ps, Pl = trans.divideP(Prcp, T1, T2)

iT = 0
Sf, qf = self.forwardLong(Sf, Ps[iT, ...], fm[iT, ...])
