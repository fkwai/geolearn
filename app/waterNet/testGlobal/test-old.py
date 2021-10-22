
from hydroDL.model import trainBasin
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

dataName = 'HBN_Q90ref'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = gridMET.varLst
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['runoff']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'B10'
testSet = 'A10'
DM = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM.trans(mtdXC=mtdXC)
dataTup = DM.getData()
dataTupRaw = DM.getDataRaw()

# random subset
batchSize = [365, 10]
sizeLst = trainBasin.getSize(dataTup)
[x, xc, y, yc] = dataTup
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
xT = torch.from_numpy(xTemp).float()
xcT = torch.from_numpy(xcTemp).float()
yT = torch.from_numpy(yTemp).float()
ycT = torch.from_numpy(ycTemp).float()

# water net
nh = 8
ns = nbatch
ng = len(varXC)
S0 = torch.zeros(ns, nh)
H0 = torch.zeros(ns, nh)
Yout = torch.zeros(nt, ns)


# inputs
T1 = xT[:, :, varX.index('tmmn')]-273.15
T2 = xT[:, :, varX.index('tmmx')]-273.15
Ta = (T1+T2)/2
P = xT[:, :, varX.index('pr')]
E = xT[:, :, varX.index('etr')]
rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
rP[T1 >= 0] = 1
rP[T2 <= 0] = 0
Ps = (1-rP)*P
Pl = rP*P

# weights
modelG = nn.Linear(ng, nh*4)
w = modelG(xcT)
gm = torch.exp(w[:, :nh])+1
ge = torch.sigmoid(w[:, nh:nh*2])
go = torch.sigmoid(w[:, nh*2:nh*3])
ga = torch.softmax(w[:, nh*3:], dim=1)

# iteration
k = 0
Sm = torch.minimum(S0, torch.relu(Ta[k, :, None]*gm))
S = S0+Ps[k, :, None]
H = H0+Sm+Pl[k, :, None]
Q = H*go
H0 = H-Q
S0 = S-Sm
Y = torch.sum(Q*ga, dim=1)
Yout[k, :]=Y
