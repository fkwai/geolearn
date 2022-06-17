
from sklearn.decomposition import PCA
import sklearn
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS
from hydroDL import utils
importlib.reload(waterNetTestC)


# extract data
dataName = 'weaG200all'
# def train(dataName, nm, codeLst):
DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+DF.varC
mtdY = ['skip'] + ['scale' for code in DF.varC]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()
DM0 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM0.borrowStat(DM1)
dataTup0 = DM0.getData()

# check data plot
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
ng = xc.shape[-1]
nh = 16
nr = 5
nc = len(DF.varC)
nm = nh
[rho, nbatch] = batchSize

model = waterNetTestC.Wn0119EM(nh, ng, nr, nc, nm).cuda()
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTempEM'
modelFile = 'wnem0119-{}-ep{}'.format(dataName, 500)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))

# test
model.eval()

[x, xc, y, yc] = dataTup2
t = DF.getT(testSet)
nt, ns, _ = y.shape

xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()

testBatch = 20
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
yP = np.ndarray([nt-nr+1, ns, nc+1])
for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    yP[:, iS[k]:iE[k], :] = yOut.detach().cpu().numpy()
model.zero_grad()
qP = yP[:, :, 0]
cP = yP[:, :, 1:]

utils.stat.calCorr(qP, y[nr-1:, :, 0])
for k in range(nc):
    a = utils.stat.calCorr(cP[:, :, k], y[nr-1:, :, k+1])
    np.nanmean(a)

iP = 5
ic = 0
fig, axes = plt.subplots(nc, 1)
for ic in range(nc):
    axplot.plotTS(axes[ic], t[nr-1:], [y[nr-1:, iP, ic+1],
              yP[:, iP, ic+1]], cLst='kr')
fig.show()
