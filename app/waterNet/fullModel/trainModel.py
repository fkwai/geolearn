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

model.train()
for ep in range(1, 1001):
    t0 = time.time()
    model.zero_grad()
    optim.zero_grad()
    yOut = model(x, xc)    
    t1 = time.time()
    loss = lossFun(yOut[:, :, None], y[rhoW+nr - 1 :, :, :])
    # loss = lossFun(yOut[:, :, None], y)
    print('forward {:.2f}'.format(t1 - t0))    
    loss.backward()
    t2 = time.time()
    print('backward {:.2f}'.format(t2 - t1))
    optim.step()
    print(ep, loss.item())
    if ep % 50 == 0:
        modelFile = os.path.join(
            saveDir, 'wfq-{}-ep{}'.format(dataName, ep))
        torch.save(model.state_dict(), modelFile)

# Qpr = torch.stack(Qp)
# Qsr = torch.stack(Qs)
# Qdr = torch.stack(Qd)
# Hfr = torch.stack(Hf)
# Hsr = torch.stack(Hs)
# Hdr = torch.stack(Hd)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(yOut[:, 0].detach().numpy())
fig.show()

# Qp, Qs, Qd = [], [], []
# Hf, Hs, Hd = [], [], []
# Qp, Qs, Qd, Hf, Hs, Hd = [1, 1, 1, 1, 1, 1]
# import make_dot
