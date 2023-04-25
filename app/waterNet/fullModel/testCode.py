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

# import importlib
# import hydroDL.model.waterNet.modelFull

# importlib.reload(hydroDL.model.waterNet.modelFull)
model = WaterNet0313(nf, ng, nh, nr, rho=rho)
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D()

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    xc = xc.cuda()
    y = y.cuda()
    lossFun = lossFun.cuda()

class test(torch.nn.Module):
    def __init__(self, nf, ng, nh, nr, rho=(5, 365, 0), hs=256, dr=0.5):
        super(test, self).__init__()
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
            gl=lambda x: torch.pow(torch.sigmoid(x) * 10, 3),  # effective depth
            qb=lambda x: torch.relu(x) / 10,  # baseflow
            ga=lambda x: torch.softmax(x, -1),  # area
            gi=lambda x: F.hardsigmoid(x) / 2,  # interception
            ge=lambda x: torch.relu(x),  # evaporation
        )
        self.kDict = dict(
            km=lambda x: torch.exp(x),  # snow melt
        )
        self.FC = nn.Linear(self.ng, hs)
        self.FC_r = nn.Linear(hs, self.nh * (self.nr + 1))
        self.FC_g = nn.Linear(hs, self.nh * len(self.gDict))
        self.FC_kin = nn.Linear(4, hs)
        self.FC_kout = nn.Linear(hs, self.nh * len(self.kDict))
        self.reset_parameters()

    def getParam(self, x, xc):
        f = x[:, :, 2:]  # T1, T2, Rad and Hum
        nt = x.shape[0]
        state = self.FC(xc)
        mask_k = createMask(state, self.dr)
        mask_g = createMask(state, self.dr)
        mask_r = createMask(state, self.dr)
        pK = self.FC_kout(
            DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k, self.training)
        )  # check in debug
        pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g, self.training))
        pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r, self.training))
        paramK = sepParam(pK, self.nh, self.kDict)
        paramG = sepParam(pG, self.nh, self.gDict)
        paramR = func.onePeakWeight(pR, self.nh, self.nr)
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

# forward    
self=test(nf, ng, nh, nr, rho=(5, 365, 10), hs=256, dr=0.5)
nt = x.shape[0]
ns = x.shape[1]
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
storage = self.initState(ns)
paramK, paramG, paramR = self.getParam(x, xc)
Ps, Pl = func.divideP(Prcp, T1, T2)
Ps = Ps.unsqueeze(-1)
Pl = Pl.unsqueeze(-1)
Evp = Evp.unsqueeze(-1)
input = [Ps, Pl, Evp]
param = [paramK, paramG, paramR]

t0 = time.time()
SfLst, SsLst, SgLst = [], [], []
with torch.no_grad():
    for iT in range(nt):
        storage, flux = bucket.step(iT, storage, input, param)
        Sf, Ss, Sd = storage
        SfLst.append(Sf)
        SsLst.append(Ss)
        SgLst.append(Sd)
time.time()-t0

t0 = time.time()
for iT in range(nt):
    storage, flux = bucket.step(iT, storage, input, param)
    Sf, Ss, Sd = storage
    if (iT - self.rho_warmup) % self.rho_long == 0:
        Sf=Sf.detach()
    SfLst.append(Sf)
    SsLst.append(Ss.detach())
    SgLst.append(Sd.detach())
time.time()-t0
