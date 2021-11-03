
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetGlobal
from hydroDL.master import basinFull
import importlib

importlib.reload(waterNetGlobal)
importlib.reload(crit)

dataName = 'QN90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm']
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
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
ng = len(varXC)
ns = len(DF.siteNoLst)

model = waterNetGlobal.WaterNet3(nh, 1, ng)
model = model.cuda()

sn = 1e-8
# random subset
ns = len(DF.siteNoLst)
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
[rho, nbatch] = batchSize

# nIterEp = int(np.ceil(np.log(0.01)/np.log(1 - nbatch*rho/2000/nt)))
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
lossLst = list()
saveDir = r'/scratch/users/kuaifang/temp/'

saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'model-{}-ep{}'.format('QN90ref', 180)

model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))

model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
t = DF.getT(testSet)
yOut, (q1Out, q2Out, q3Out) = model(xP, xcP, outQ=True)
model.zero_grad()

# w = model.fc(xcP)
yP = yOut.detach().cpu().numpy()
q1P = q1Out.detach().cpu().numpy()
q2P = q2Out.detach().cpu().numpy()
q3P = q3Out.detach().cpu().numpy()

# LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=1000)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, y[:, :, 0])
corr1 = utils.stat.calCorr(yP, y[:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])

np.mean(corr2)

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(yP[:, k], 'r')
ax.plot(yL[:, k], 'b')
ax.plot(y[:, k, 0], 'k')
fig.show()

fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['nash', 'corr'],
                            label2=['waternet2', 'LSTM'])
fig.show()
