
from tkinter import X
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

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
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
nr = 5
model = waterNet.WaterNet0119(nh, len(varXC), nr)
model = model.cuda()

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0119-{}-ep{}'.format('QN90ref', 300)
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
yP = np.ndarray([nt-nr+1, ns])
for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
model.zero_grad()


# LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, y[nr-1:, :, 0])
corr1 = utils.stat.calCorr(yP, y[nr-1:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])

lat, lon = DF.getGeo()


def funcM():
    figM = plt.figure(figsize=(12, 5))
    gsM = gridspec.GridSpec(1, 1)
    # axM2 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash2-nash1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat, lon, nash2-nash1, centerZero=True)
    axM.set_title('LSTM - waterNet Nash')
    axM = np.array([axM])
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    axT = axP.twinx()
    return figM, axM, figP, np.array([axP, axT]), lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = DF.siteNoLst[iP]
    t = DF.getT(testSet)
    legLst = ['obs',
              'waterNet {:.2f} {:.2f}'.format(nash1[iP], corr1[iP]),
              'LSTM {:.2f} {:.2f}'.format(nash2[iP], corr2[iP])
              ]
    axplot.plotTS(axP[1], t[nr-1:], [x[nr-1:, iP, 0]],
                  lineW=[0.5], cLst='c', legLst=['prcp'])
    axplot.plotTS(axP[0], t[nr-1:], [y[nr-1:, iP, 0], yP[:, iP], yL[nr-1:, iP]],
                  lineW=[2, 1, 1], cLst='krb', legLst=legLst)
    axP[1].invert_yaxis()
    strTitle = ('{}'.format(DF.siteNoLst[iP]))
    axP[0].set_title(strTitle)


figM, figP = figplot.clickMap(funcM, funcP)


matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})


# 08101000
# 10172800
# 02053200
# 03180500

P = DF.f[:, :, 0]
Q = DF.q[:, :, 1]
E = DF.q[:, :, 6]


Pm = np.nanmean(P, axis=0)
Qm = np.nanmean(Q, axis=0)
figM = plt.figure(figsize=(12, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, (Pm-Qm)/Em)
figM.show()

figM = plt.figure(figsize=(12, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, np.nanstd(Q,axis=0)/np.nanmean(Q,axis=0))
figM.show()

figM = plt.figure(figsize=(12, 5))
gsM = gridspec.GridSpec(1, 1)
# axM2 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash2-nash1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, corr2**2-corr1**2, centerZero=True)
axM.set_title('LSTM - waterNet Nash')
figM.show()


fig, ax = plt.subplots(1, 1)
ax.plot(Qm/Pm, corr1, '*')
ax.set_ylim(0,1)
fig.show()
