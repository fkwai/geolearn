
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

outDir = os.path.join(saveDir, 'HBN36', 'wn2-ep1000-save')
modelFile = os.path.join(outDir, 'model')
model.load_state_dict(torch.load(modelFile))

model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
t = DF.getT(testSet)
# yP = model(xP, xcP).detach().cpu().numpy()
yOut, (q1Out, q2Out, q3Out) = model(xP, xcP, outQ=True)
w = model.fc(xcP)
gaOut = torch.softmax(w[:, nh*6:nh*7], dim=1)

yP = yOut.detach().cpu().numpy()
q1P = q1Out.detach().cpu().numpy()
q2P = q2Out.detach().cpu().numpy()
q3P = q3Out.detach().cpu().numpy()
ga = gaOut.detach().cpu().numpy()
model.zero_grad()

# LSTM
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(outName, DF=DF, testSet=testSet, reTest=True)
yL = yL[:, :, 0]


nash1 = utils.stat.calNash(yP, y[:, :, 0])
corr1 = utils.stat.calCorr(yP, y[:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])
fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['nash', 'corr'],
                            label2=['waternet2', 'LSTM'])
fig.show()

fig, axes = plt.subplots(2, 1)
axplot.plot121(axes[0], nash1, nash2)
axplot.plot121(axes[1], corr1, corr2)
fig.show()

# saveSubDir = os.path.join(outDir, 'qPlot')
# if ~os.path.exists(saveSubDir):
#     os.makedirs(saveSubDir)
# for k in range(ns):
#     fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
#     ax = axes[0]
#     ax.plot(t, y[:, k], '-k', linewidth=2)
#     ax.plot(t, yP[:, k], '-r', linewidth=1,
#             label='waternet2 {:.2f} {:.2f}'.format(nash1[k], corr1[k]))
#     ax.plot(t, yL[:, k], '-b', linewidth=1,
#             label='LSTM {:.2f} {:.2f}'.format(nash2[k], corr2[k]))
#     ax.set_title('{}'.format(DF.siteNoLst[k]))
#     ax.legend()
#     for ax, qP in zip(axes[1:], [q1P, q2P, q3P]):
#         ax.plot(t, y[:, k], '-k', linewidth=2)
#         ax.plot(t, qP[:, k])
#     fig.subplots_adjust(hspace=0)
#     fig.show()
#     fig.savefig(os.path.join(saveSubDir, DF.siteNoLst[k]))

