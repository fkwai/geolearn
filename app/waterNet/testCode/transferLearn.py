
from hydroDL.post import axplot, figplot, mapplot
from hydroDL.master import basinFull
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
from hydroDL import utils
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from hydroDL.model.waterNet import convTS, sepPar
from hydroDL.model import waterNetTest

import pandas as pd
import os

dataName = 'B5Y09-00955'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = ['00955']
nc = len(codeLst)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip' for k in range(nc+1)]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

# model
nh = 16
nr = 5
model = waterNetTest.WaterNet0110(nh, len(varXC), nr)
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
sizeLst = trainBasin.getSize(dataTup1)
[rho, nbatch] = batchSize

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0110Q-00955-{}-ep{}'.format(dataName, 1000)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))


class test(waterNetTest.WaterNet0110):
    def __init__(self, nc, dictState=None):
        super().__init__(nh, len(varXC), nr)
        if dictState is not None:
            super().load_state_dict(dictState)
        self.nc = nc
        # [eqs,eqg]
        self.cLst = ['exp', 'exp']
        self.fcC = nn.Sequential(
            nn.Linear(self.ng, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, nc*nh*len(self.cLst)))
        # [rs,rg]
        self.ctLst = ['sigmoid', 'sigmoid']
        self.fcCT = nn.Linear(2+self.ng, nc*nh*len(self.cLst))
        # ISSUE - no reset parameters - and the old one not work neither


dictState = torch.load(os.path.join(saveDir, modelFile))
aa = test(1, dictState=dictState)

# aa.state_dict().keys()

aa.state_dict()['fcC.0.weight']
# aa.state_dict()['fcR.0.weight']
# model.state_dict()['fcR.0.weight']

for layer in model.children():
    print(layer)
    if hasattr(layer, 'reset_parameters'):
        print(layer)
