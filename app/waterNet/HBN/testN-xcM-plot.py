
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
varX = ['pr', 'etr', 'tmmn', 'tmmx']
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['runoff']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['stan' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'B10'
testSet = 'A10'
DM = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM.trans(mtdXC=mtdXC)
dataTup = DM.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.trans(mtdXC=mtdXC)
dataTup2 = DM2.getData()
DM3 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM3.trans(mtdXC=mtdXC)
dataTup3 = DM3.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
ng = ns
model = waterNetGlobal.WaterNet2(nh, ng)
model = model.cuda()
optim = torch.optim.RMSprop(model.parameters())
optim = torch.optim.Adadelta(model.parameters())
lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogAll2D().cuda()

sn = 1e-8
# random subset
# madeup xc
ns = len(DF.siteNoLst)
# xcM = np.random.random([ns, ng])
xcM = np.eye(ns)

outDir = os.path.join(saveDir, 'HBN36', 'wn2-ep2000')
modelFile = os.path.join(outDir, 'model')
model.load_state_dict(torch.load(modelFile))


model.eval()
[x, xc, y, yc] = dataTup2
xc = xcM
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
t = DF.getT(testSet)
yP = model(xP, xcP).detach().cpu().numpy()
model.zero_grad()

# LSTM
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(outName, DF=DF, testSet=testSet, reTest=False)
yL = yL[:, :, 0]

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
