
import matplotlib.dates as mdates
from builtins import anext
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
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\waterNet'


siteNoPlot = ['08101000',
              '10172800',
              '02053200',
              '03180500']
tRLst = [[20120501, 20130501],
         [20110101, 20120101],
         [20140101, 20140601],
         [20180601, 20181201]]

for siteNo, tR in zip(siteNoPlot, tRLst):
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 5)
    ngs = 3
    iP = DF.siteNoLst.index(siteNo)
    legLst = ['obs',
              'waterNet {:.2f} {:.2f}'.format(nash1[iP], corr1[iP]),
              'LSTM {:.2f} {:.2f}'.format(nash2[iP], corr2[iP])]
    t = DF.getT(testSet)
    ax = fig.add_subplot(gs[0, :ngs])
    axplot.plotTS(ax, t[nr-1:], [y[nr-1:, iP, 0], yP[:, iP], yL[nr-1:, iP]],
                  lineW=[2, 1, 1], cLst='krb', legLst=legLst)

    sd = utils.time.t2dt(tR[0])
    ed = utils.time.t2dt(tR[1])
    indT1 = np.where(t == utils.time.t2dt(tR[0]))[0][0]
    indT2 = np.where(t == utils.time.t2dt(tR[1]))[0][0]
    axT = fig.add_subplot(gs[0, ngs:])
    dataT = [y[indT1:indT2, iP, 0],
             yP[indT1-nr+1:indT2-nr+1, iP], yL[indT1:indT2, iP]]
    axplot.plotTS(axT, t[indT1:indT2], dataT,
                  lineW=[2, 1, 1], cLst='krb')
    axP = axT.twinx()
    axplot.plotTS(axP, t[indT1:indT2], [x[indT1:indT2, iP, 0]],
                  lineW=[1], cLst='c', legLst=['prcp'])
    axP.invert_yaxis()
    axT.set_xticks([sd, ed])
    fig.show()
    fig.savefig(os.path.join(dirPaper, 'ts_{}'.format(siteNo)))


siteNoPlot = ['08101000',
              '10172800',
              '02053200',
              '03180500']
fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 7)
# axM2 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash2-nash1)
axM = mapplot.mapPoint(
    fig, gs[0, :5], lat, lon, nash1-nash2, centerZero=True)
axM.set_title('waterNet NSE - LSTM NSE')
for siteNo in siteNoPlot:
    xLoc = lon[DF.siteNoLst.index(siteNo)]
    yLoc = lat[DF.siteNoLst.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='black', fill=False)
    axM.add_patch(circle)
ax = fig.add_subplot(gs[0, 5:])
ax.plot(nash1, nash2, 'o')
ax.plot([-0.25, 1], [-0.25, 1], '-k')
ax.set_ylim(-0.25, 1)
ax.set_xlim(-0.25, 1)
ax.set_xlabel('waterNet NSE')
ax.set_ylabel('LSTM NSE')
ax.set_aspect(1)
for siteNo in siteNoPlot:
    xLoc = nash1[DF.siteNoLst.index(siteNo)]
    yLoc = nash2[DF.siteNoLst.index(siteNo)]
    ax.plot(xLoc, yLoc, 'ro')

fig.show()
fig.savefig(os.path.join(dirPaper, 'mapDiff'))
