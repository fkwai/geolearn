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
nh = 4
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

modelFile = os.path.join(saveDir, 'wfq-{}-ep{}'.format(dataName, 500))
model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))

model.eval()
yOut = model(x, xc)
corr = utils.stat.calCorr(yOut.detach().numpy(), yP[nr - 1 :, :, 0])
np.nanmean(corr)
