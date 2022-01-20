
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL.master import basinFull
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
from hydroDL import utils

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
model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
nt, ns, _ = y.shape
t = DF.getT(testSet)
testBatch = 100
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
qP = np.ndarray([nt-nr+1, ns])
cP = np.ndarray([nt-nr+1, ns])

for k in range(len(iS)):
    print('batch {}'.format(k))
    qOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    qP[:, iS[k]:iE[k]] = qOut.detach().cpu().numpy()
model.zero_grad()

nashQ1 = utils.stat.calNash(qP, y[nr-1:, :, 0])
corrQ1 = utils.stat.calCorr(qP, y[nr-1:, :, 0])


# LSTM
outName = 'B5Y09-00955-Q'
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=1000)
qL = yL[:, :, 0]
cL = yL[:, :, 1]

nashQ2 = utils.stat.calNash(qL, y[:, :, 0])
corrQ2 = utils.stat.calCorr(qL, y[:, :, 0])

fig, axes = figplot.boxPlot([[corrQ1, corrQ2]],
                            label1=['Q'],
                            label2=['WN0110', 'LSTM'])
fig.show()

siteNo = '06317000'
indS = DF.siteNoLst.index(siteNo)
fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(t[nr-1:], qP[:, indS], '-r')
ax.plot(t, y[:, indS, 0], '-k')
fig.show()
