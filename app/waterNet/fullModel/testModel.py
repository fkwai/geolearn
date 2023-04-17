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
from hydroDL import kPath, utils


# load data
saveDir = os.path.join(kPath.dirWQ, 'waterNet')
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
testSet = 'all'
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

# (xP, xcP, yP, ycP) = dataTup1
(xP, xcP, yP, ycP) = dataTup2
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
model = WaterNet0313(nf, ng, nh, nr, rho=(5, 365, rhoW))
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D()

if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    xc = xc.cuda()
    y = y.cuda()
    lossFun = lossFun.cuda()

modelFile = os.path.join(saveDir, 'wfq2-{}-ep{}'.format(dataName, 500))

model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))

# model.eval()
t0 = time.time()
yOut, (Qf, Qp, Qs, Qd), (Hf, Hs, Hd) = model(x, xc, outStep=True)
time.time() - t0

Hf, Hs, Hd = [l.detach().numpy() for l in [Hf, Hs, Hd]]
Qp, Qs, Qd = [l.detach().numpy() for l in [Qp, Qs, Qd]]
Q = yOut.detach().numpy()
obs = yP[rhoW + nr - 1 :, :, 0]
corr = utils.stat.calCorr(Q, obs)
np.nanmean(corr)


# parameters
paramK, paramG, paramR = model.getParam(x, xc)

# plot
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec

lat, lon = DF.getGeo()
t = DF.t[rhoW + nr - 1 :]
t2 = DF.t[nr - 1 :]

# result
def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, corr, s=16, cb=True)
    figP, axP = plt.subplots(4, 1, figsize=[15, 8], sharex=True)
    figP.subplots_adjust(wspace=0, hspace=0)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP, DF.siteNoLst[iP])
    axP[0].plot(t, obs[:, iP], 'k-')
    axP[0].plot(t, Q[:, iP], 'r-')
    # axP[1].plot(t, Qp[:, iP, :])
    # axP[2].plot(t, Qs[:, iP, :])
    # axP[3].plot(t, Qd[:, iP, :])
    axP[1].plot(t2, Hf[nr - 1 :, iP, :])
    axP[2].plot(t2, Hs[nr - 1 :, iP, :])
    axP[3].plot(t2, Hd[nr - 1 :, iP, :])


figM, figP = figplot.clickMap(funcM, funcP)

# inputs
iP = 0
Prcp, Evp, T1, T2, Rad, Hum = [x[:, :, k] for k in range(x.shape[-1])]
Ps, Pl = func.divideP(Prcp, T1, T2)
Ps = Ps.unsqueeze(-1)
Pl = Pl.unsqueeze(-1)
Evp = Evp.unsqueeze(-1)
Is = Qf + Pl * paramG['gi'] - Evp * paramG['ge']
# Is = Qf + Pl * paramG['gi']


def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, corr, s=16, cb=True)
    figP, axP = plt.subplots(4, 1, figsize=[15, 8], sharex=True)
    figP.subplots_adjust(wspace=0, hspace=0)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP, DF.siteNoLst[iP])
    axP[0].plot(t, obs[:, iP], 'k-')
    axP[0].plot(t, Q[:, iP], 'r-')
    axP[1].plot(t2, Is[nr - 1 :, iP, :].detach().numpy())
    axP[2].plot(t2, Hs[nr - 1 :, iP, :])
    axP[3].plot(t2, Hd[nr - 1 :, iP, :])


figM, figP = figplot.clickMap(funcM, funcP)


paramG.keys()
a = paramG['ge'].detach().numpy()
fig, ax = plt.subplots(1, 1)
im = ax.imshow(a)
# im=ax.imshow(a,vmax=200)
fig.colorbar(im, ax=ax)
fig.show()

fig, ax = plt.subplots(1, 1)
ax.hist(a.flatten(), bins=10)
fig.show()
