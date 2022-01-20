
import os
import copy
import collections
from gc import collect
from hydroDL.model.waterNet import convTS, sepPar
from hydroDL.model import trainBasin, crit
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

from hydroDL.model import waterNetTestC
import importlib

importlib.reload(waterNetTestC)
importlib.reload(crit)

dataName = 'B5Y09-00955'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = ['00955']
nc = len(codeLst)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
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
# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0110Q-00955-{}-ep{}'.format(dataName, 1000)
dictQ = torch.load(os.path.join(saveDir, modelFile))

model = waterNetTestC.Trans0110C2(nh, len(varXC), nr, nc, dictQ=dictQ)
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

model.eval()
t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
qOut, cOut = model(xP, xcP)
qP = qOut.detach().cpu().numpy()
cP = cOut.detach().cpu().numpy()
lossQ = lossFun(qOut, yT[nr-1:, :, 0])
lossC = lossFun(cOut[:, :, 0], yT[nr-1:, :, 1])
print(lossQ.item(), lossC.item())

siteNo = '06317000'
indS = DF.siteNoLst.index(siteNo)
fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(t[nr-1:], qP[:, indS], '-r')
ax.plot(t, y[:, indS, 0], '-k')
fig.show()
