
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
# yP = np.ndarray([nt-nr+1, ns])
# for k in range(len(iS)):
#     print('batch {}'.format(k))
#     yOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]],)
#     yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
yOut, (QpR, QsR, QgR), (SfT, SsT, SgT) = model(xP, xcP, outStep=True)
model.zero_grad()
yP = yOut.detach().cpu().numpy()
Qp = QpR.detach().cpu().numpy()
Qs = QsR.detach().cpu().numpy()
Qg = QgR.detach().cpu().numpy()
Sf = SfT.detach().cpu().numpy()
Ss = SsT.detach().cpu().numpy()
Sg = SgT.detach().cpu().numpy()


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

# ts
siteNoSel = ['03173000', '13235000', '08377900', '06885500']
indSLst = [DF.siteNoLst.index(s) for s in siteNoSel]
dataPlot = list()
dataPlot = [y[nr-1:, indSLst, 0], yP[:, indSLst], yL[nr-1:, indSLst]]
fig, axes = figplot.multiTS(
    t[nr-1:], dataPlot, labelLst=siteNoSel)
fig.show()
# map
# circle = plt.Circle([xLoc[iP], yLoc[iP]], 1,
#                     color='black', fill=False)
# ax.add_patch(circle)
figM = plt.figure()
gsM = gridspec.GridSpec(2, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1, vRange=[0, 1], s=10)
axM0.set_title('waterNet Nash')
axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, nash2, vRange=[0, 1], s=10)
axM1.set_title('LSTM Nash')
figM.show()

# site 09163500
siteNo = '06885500'
indS = DF.siteNoLst.index(siteNo)
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
dataPlot = [y[nr-1:, indS, 0], yP[:, indS], yL[nr-1:, indS]]
legLst = ['obs', 'waterNet {:.2f}'.format(nash1[indS]),
          'LSTM {:.2f}'.format(nash2[indS])]
axplot.plotTS(ax, t[nr-1:], dataPlot, legLst=legLst)
fig.show()
# Q
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
axes[0].plot(t[nr-1:], Qp[:, indS, :])
axes[1].plot(t[nr-1:], Qs[:, indS, :])
axes[2].plot(t[nr-1:], Qg[:, indS, :])
fig.subplots_adjust(hspace=0)
fig.show()

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
axes[0].plot(t, Sf[:, indS, :])
axes[1].plot(t, Ss[:, indS, :])
axes[2].plot(t, Sg[:, indS, :])
fig.subplots_adjust(hspace=0)
fig.show()
