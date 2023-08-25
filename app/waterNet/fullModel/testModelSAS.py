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
from hydroDL.model.waterNet.modelSAS import WaterNet0510
from hydroDL.model import crit
import time
import os
import matplotlib.pyplot as plt


# load data
saveDir = r'/oak/stanford/schools/ees/kmaher/Kuai/waterQuality/waterNet/'
saveDir = r'/home/kuai/work/waterQuality/waterNet/'
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
model = WaterNet0510(nf, ng, nh)

modelFile = os.path.join(saveDir, 'wnSAS-{}-ep{}'.format(dataName, 4))

model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))

# check param predicted
paramK, paramG = model.getParam(x, xc)

gk = paramG['gk'].detach().numpy()
gl = paramG['gl'].detach().numpy()

fluxLst, input, param, SfLst, DLst = model.forwardAll(x, xc, outOpt=1)

model.eval()
self = model
f = x[:, :, 2:]  # T1, T2, Rad and Hum
nt = x.shape[0]
state = self.FC(xc)
mask_k = createMask(state, self.dr)
mask_g = createMask(state, self.dr)
# mask_r = createMask(state, self.dr)
pK = self.FC_kout(
    DropMask.apply(torch.tanh(self.FC_kin(f) + state), mask_k, self.training)
)
pG = self.FC_g(DropMask.apply(torch.tanh(state), mask_g, self.training))
# pR = self.FC_r(DropMask.apply(torch.tanh(state), mask_r, self.training))
paramK = sepParam(pK, self.nh, self.kDict)
paramG = sepParam(pG, self.nh, self.gDict)

fig, ax = plt.subplots(1, 1)
im = ax.imshow(pG.detach().numpy())
fig.colorbar(im, ax=ax)
fig.show()

nt = len(fluxLst)
Q = torch.stack([fluxLst[x][50, 8, :] for x in range(nt)]).numpy()

fig, ax = plt.subplots(1, 1)
ax.plot(Q)
fig.show()

8 * 13514 * 366 * 16 * 500 / 1024 / 1024
