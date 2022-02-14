
from hydroDL.model.waterNet import convTS, sepPar
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
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

nr = 5
model = waterNetTest.WaterNet0119(nh, len(varXC), nr)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0119-{}-ep{}'.format('QN90ref', 100)
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
yP1 = np.ndarray([nt-nr+1, ns, nh])
yP2 = np.ndarray([nt-nr+1, ns, nh])
yP3 = np.ndarray([nt-nr+1, ns, nh])

for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut, (Q1out, Q2out, Q3out) = model(
        xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]], outStep=True)
    yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
    yP1[:, iS[k]:iE[k]] = Q1out.detach().cpu().numpy()
    yP2[:, iS[k]:iE[k]] = Q2out.detach().cpu().numpy()
    yP3[:, iS[k]:iE[k]] = Q3out.detach().cpu().numpy()
    model.zero_grad()

# model weight
x = xP
xc = xcP
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
nh = model.nh
nr = model.nr
xcT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
w = model.fcW(xc)
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, model.wLst)
gL = gL**2
kg = kg/10
ga = torch.softmax(ga, dim=1)
v = model.fcT(xcT)
[vi, ve, vm] = sepPar(v, nh, model.vLst)
vi = F.hardsigmoid(v[:, :, :nh]*2)
ve = ve*2
wR = model.fcR(xc)
vf = torch.arccos((T1+T2)/(T2-T1))/3.1415
vf[T1 >= 0] = 0
vf[T2 <= 0] = 1

# plot attributes
a = ga.detach().cpu().numpy()
q3 = np.sum(yP3*a, axis=2)
kPlot = np.nansum(q3,axis=0)/np.nansum(yP,axis=0)
kPlot = torch.sum(qb*ga, dim=1).detach().cpu().numpy()

lat, lon = DF.getGeo()
figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, kPlot)
figM.show()


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
xClick = -107.02817413764274
yClick = 38.90991781866984
iP = np.nanargmin(np.sqrt((xClick - lon)**2 + (yClick - lat)**2))


importlib.reload(mapplot)
importlib.reload(axplot)


def funcM():
    figM = plt.figure(figsize=(12, 5))
    gsM = gridspec.GridSpec(1, 3)
    axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1)
    axM0.set_title('waterNet Nash')
    axM1 = mapplot.mapPoint(figM, gsM[0, 1], lat, lon, nash2)
    axM1.set_title('LSTM Nash')
    axM2 = mapplot.mapPoint(figM, gsM[0, 2], lat, lon, nash2-nash1)
    axM2.set_title('LSTM - waterNet Nash')
    axM = np.array([axM0, axM1, axM2])
    figP, axP = plt.subplots(4, 1, figsize=(12, 4))
    figP.subplots_adjust(hspace=0)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = DF.siteNoLst[iP]
    t = DF.getT(testSet)
    legLst = ['obs',
              'waterNet {:.2f} {:.2f}'.format(nash1[iP], corr1[iP]),
              'LSTM {:.2f} {:.2f}'.format(nash2[iP], corr2[iP])
              ]
    axplot.plotTS(axP[0], t[nr-1:], [y[nr-1:, iP, 0], yP[:, iP], yL[nr-1:, iP]],
                  lineW=[2, 1, 1], cLst='krb', legLst=legLst)
    axP[1].plot(t[nr-1:], yP1[:, iP, :])
    axP[2].plot(t[nr-1:], yP2[:, iP, :])
    axP[3].plot(t[nr-1:], yP3[:, iP, :])
    strTitle = ('{}'.format(DF.siteNoLst[iP]))
    axP[0].set_title(strTitle)


figM, figP = figplot.clickMap(funcM, funcP)


fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['nash', 'corr'],
                            label2=['waternet0119', 'LSTM'],
                            yRange=[0, 1])
fig.show()

figM = plt.figure()
gsM = gridspec.GridSpec(2, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1, vRange=[0, 1])
axM0.set_title('waterNet Nash')
axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, nash2, vRange=[0, 1])
axM1.set_title('LSTM Nash')
figM.show()


figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat,
                        lon, nash2-nash1, vRange=[-0.2, 0.2])
axM0.set_title('LSTM Nash - waterNet Nash')
figM.show()
