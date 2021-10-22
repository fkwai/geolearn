
import os
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

from hydroDL.model import waterNetGlobal
import importlib

importlib.reload(waterNetGlobal)
importlib.reload(crit)
saveDir = r'C:\Users\geofk\work\temp\waternet'

dataName = 'HBN_Q90ref'
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
DM3 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM3.borrowStat(DM1)
dataTup3 = DM3.getData()

np.where(np.isnan(dataTup1[0]))
np.where(np.isnan(dataTup1[1]))
np.where(np.isnan(dataTup1[2]))
# np.where(np.isnan(dataTup[3]))

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)

model = waterNetGlobal.WaterNet3(nh, 1, ng)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

sn = 1e-8
# random subset
ns = len(DF.siteNoLst)
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
model.train()
for kk in range(1000):
    batchSize = [1000, 100]
    [rho, nbatch] = batchSize
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
    xT = torch.from_numpy(xTemp).float().cuda()
    xcT = torch.from_numpy(xcTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    ycT = torch.from_numpy(ycTemp).float().cuda()
    model.zero_grad()
    yP = model(xT, xcT)
    loss = lossFun(yP[:, :, None], yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss.item())
    # w = model.modelG2(torch.sigmoid(model.modelG1(xcT)))
    # print(w.max(), w.min())

model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
t = DF.getT(testSet)
yP = model(xP, xcP).detach().cpu().numpy()
model.zero_grad()

# LSTM
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(outName, DF=DF, testSet=testSet, reTest=True)
yL = yL[:, :, 0]


outDir = os.path.join(saveDir, 'HBN36', 'wn2-ep2000')
modelFile = os.path.join(outDir, 'model')
nash1 = utils.stat.calNash(yP, y[:, :, 0])
corr1 = utils.stat.calCorr(yP, y[:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])
fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['nash', 'corr'],
                            label2=['waternet2', 'LSTM'])
fig.savefig(os.path.join(outDir, 'box'))
fig.show()

fig, axes = plt.subplots(2, 1)
axplot.plot121(axes[0], nash1, nash2)
axplot.plot121(axes[1], corr1, corr2)
fig.savefig(os.path.join(outDir, 'WNvsLSTM'))
fig.show()

if ~os.path.exists(outDir):
    os.makedirs(outDir)
for k in range(ns):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t, y[:, k], '-k', linewidth=2)
    ax.plot(t, yP[:, k], '-r', linewidth=1,
            label='waternet2 {:.2f} {:.2f}'.format(nash1[k], corr1[k]))
    ax.plot(t, yL[:, k], '-b', linewidth=1,
            label='LSTM {:.2f} {:.2f}'.format(nash2[k], corr2[k]))
    ax.set_title('{}'.format(DF.siteNoLst[k]))
    ax.legend()
    fig.show()
    fig.savefig(os.path.join(outDir, DF.siteNoLst[k]))
torch.save(model.state_dict(), modelFile)
