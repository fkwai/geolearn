
import matplotlib
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetTest, waterNet
from hydroDL.master import basinFull
import importlib

trainSet = 'WYB09'
testSet = 'WYA09'
# dataName = 'QN90ref'
dataName = 'Q95ref'
wnName = 'WaterNet0630'
epWN = 500
epLSTM = 500
modelFile = '{}-{}-ep{}'.format(wnName, dataName, epWN)
lstmOutName = '{}-{}'.format(dataName, trainSet)
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'

# data
DF = dbBasin.DataFrameBasin(dataName)


# waterNet
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# data
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
# dataTup1 = DM1.getData()
# DM2 = dbBasin.DataModelBasin(
#     DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
# DM2.borrowStat(DM1)
# dataTup2 = DM2.getData()
DM = dbBasin.DataModelBasin(
    DF, subset='WYall', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM.borrowStat(DM1)
dataTup = DM.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
nr = 5
funcM = getattr(waterNetTest, wnName)
model = funcM(nh, len(varXC), nr)
model = model.cuda()

# water net
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))
model.eval()
[x, xc, y, yc] = dataTup

testBatch = 20
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
ns = y.shape[1]
t = DF.getT(testSet)
nt = len(t)
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
yP = np.ndarray([nt-nr+1, ns])
for k in range(len(iS)):
    print('batch {}'.format(k))
    xP = torch.from_numpy(x[:, iS[k]:iE[k], :]).float().cuda()
    xcP = torch.from_numpy(xc[iS[k]:iE[k]]).float().cuda()
    yOut = model(xP, xcP, outStep=False)
    temp = yOut.detach().cpu().numpy()
    yP[:, iS[k]:iE[k]] = temp[-nt+nr-1:, :]

# yOut, (QpR, QsR, QgR), (SfT, SsT, SgT) = model(xP, xcP, outStep=True)
# yP = yOut.detach().cpu().numpy()
# Qp = QpR.detach().cpu().numpy()
# Qs = QsR.detach().cpu().numpy()
# Qg = QgR.detach().cpu().numpy()
# Sf = SfT.detach().cpu().numpy()
# Ss = SsT.detach().cpu().numpy()
# Sg = SgT.detach().cpu().numpy()

tall = DF.getT('WYall')
_, indT, _ = np.intersect1d(tall, t, return_indices=True)
yT = y[indT, :]
# LSTM
lstmOutName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    lstmOutName, DF=DF, testSet=testSet, reTest=False, ep=epLSTM)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, yT[nr-1:, :, 0])
corr1 = utils.stat.calCorr(yP, yT[nr-1:, :, 0])
nash2 = utils.stat.calNash(yL, yT[:, :, 0])
corr2 = utils.stat.calCorr(yL, yT[:, :, 0])

lat, lon = DF.getGeo()

importlib.reload(mapplot)
importlib.reload(axplot)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# box
fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['nash', 'corr'],
                            label2=['waterNet', 'LSTM'],
                            yRange=[0, 1])
fig.show()

pred = yP
obs = yT[nr-1:, :, 0]
r = corr1
u1 = np.nanmean(pred, axis=0)
u2 = np.nanmean(obs, axis=0)
s1 = np.nanstd(pred, axis=0)
s2 = np.nanstd(obs, axis=0)
b = u1/u2
np.nanmean(b-1)
g = (s2*u1)/(s1*u2)
kge1 = 1-np.sqrt((r-1)**2+(b-1)**2+(g-1)**2)
np.mean((r-1)**2)
np.mean((b-1)**2)
np.mean((g-1)**2)
fig, ax = plt.subplots(1, 1)
ax.plot(u1, u2, '*')
fig.show()

pred = yL
obs = yT[:, :, 0]
r = corr2
u1 = np.nanmean(pred, axis=0)
u2 = np.nanmean(obs, axis=0)
s1 = np.nanstd(pred, axis=0)
s2 = np.nanstd(obs, axis=0)
b = u1/u2
np.nanmean(b-1)
g = (s2*u1)/(s1*u2)
kge2 = 1-np.sqrt((r-1)**2+(b-1)**2+(g-1)**2)
np.mean((r-1)**2)
np.mean((b-1)**2)
np.mean((g-1)**2)
fig, ax = plt.subplots(1, 1)
ax.plot(u1, u2, '*')
fig.show()
np.nanmedian(kge1)
np.nanmedian(kge2)

fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2], [kge1, kge2]],
                            label1=['nash', 'corr', 'kge'],
                            label2=['waterNet', 'LSTM'],
                            yRange=[0, 1])
fig.show()
